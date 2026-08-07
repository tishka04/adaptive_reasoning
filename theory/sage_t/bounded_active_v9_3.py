"""SAGE.T9.3 bounded-authority source-train pilot and first-progress gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v5 as t8_5
from . import live_shadow_pilot_v7 as live_i
from . import trajectory_planning_v9_2 as t9_2
from .controller import SageTConfig
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.3-bounded-active-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_3_bounded_manifest.json"
)
DEFAULT_PARENT_REPORT = t9_2.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "bounded_v9_3"
SOURCE_GAMES = ("lp85-305b61c3", "su15-4c352900")
SEED = 0
RESETS = 3
ACTIONS_PER_RESET = 50


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "bounded_active_v9_3.py": v86c._file_sha256(
            directory / "bounded_active_v9_3.py"
        )
    }


def _load_parent_report(path: str | Path = DEFAULT_PARENT_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.2 report checksum mismatch")
    if report.get("status") != "T9_2_PASSED" or report.get("t9_3_authorized") is not True:
        raise ValueError("T9.2 did not authorize T9.3")
    if report.get("bounded_authority_authorized") is not False:
        raise ValueError("bounded authority opened before T9.3 freeze")
    return report


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    parent = _load_parent_report()
    selected = str(parent["selected_challenger"])
    caps = dict(t9_2.CHALLENGERS[selected])
    t8_action = json.loads(t8_5.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_3_BOUNDED_SOURCE_TRAIN",
        "frozen_at": "2026-08-06",
        "parent_t9_2_report_checksum": parent["report_checksum"],
        "code_sha256": _code_hashes(),
        "selected_challenger": selected,
        "controller_caps": caps,
        "selected_terminal_policy": "safe_after_3",
        "source_train_games": list(SOURCE_GAMES),
        "seeds": [SEED],
        "resets": RESETS,
        "action_budget_per_reset": ACTIONS_PER_RESET,
        "runtime": dict(t8_action["runtime"]),
        "authority": {
            "mode": "bounded",
            "maximum_interventions_per_reset": 5,
            "maximum_marginal_terminal_risk": 0.05,
            "maximum_interventions_per_unknown_context": 1,
            "strong_surprise_lockout_threshold": 8.0,
            "observed_danger_is_absolute": True,
            "protected_route_is_absolute": True,
        },
        "gate": {
            "minimum_total_actions": 120,
            "minimum_real_progress_events": 1,
            "minimum_levels_completed": 1,
            "maximum_game_over_delta": 0,
            "maximum_false_high_terminal_rate": 0.05,
            "maximum_decision_p95_ms": 2500.0,
            "maximum_observation_p95_ms": 3000.0,
            "maximum_wall_seconds": 360.0,
            "maximum_illegal_actions": 0,
            "maximum_controller_errors": 0,
            "maximum_environment_errors": 0,
        },
        "firewall": {
            "source_train_only": True,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "active_authority": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.3 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_3_BOUNDED_SOURCE_TRAIN":
        raise ValueError("T9.3 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.3 code drifted")
    parent = _load_parent_report()
    if payload.get("parent_t9_2_report_checksum") != parent["report_checksum"]:
        raise ValueError("T9.3 parent report drifted")
    selected = str(parent["selected_challenger"])
    if payload.get("selected_challenger") != selected:
        raise ValueError("T9.3 challenger drifted")
    if payload.get("controller_caps") != dict(t9_2.CHALLENGERS[selected]):
        raise ValueError("T9.3 controller caps drifted")
    if tuple(payload.get("source_train_games", ())) != SOURCE_GAMES:
        raise ValueError("T9.3 source games drifted")
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
        raise ValueError("T9.3 firewall is open")
    return payload


def _first_action_risk(assessment: Any) -> float:
    value = t9_2._first_action_risk(assessment)
    return 1.0 if value is None else float(value)


class BoundedTrajectoryController(t9_2.StructuralTrajectoryController):
    """Fail-closed bounded authority using marginal first-action risk."""

    def __init__(self, *args: Any, strong_surprise_threshold: float = 8.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strong_surprise_threshold = float(strong_surprise_threshold)
        self._surprise_lockout = False
        self._bounded_by_branch: dict[int, int] = defaultdict(int)
        self._strong_surprises = 0

    def _bounded_authority(
        self,
        chosen: Any,
        *,
        context: str,
        symbolic_name: str,
        symbolic_data: Mapping[str, Any],
    ) -> tuple[str, bool]:
        if (
            chosen.first_action.action_name == str(symbolic_name).strip().upper()
            and dict(chosen.first_action.action_data) == dict(symbolic_data)
        ):
            return "bounded_agreement", False
        if self._surprise_lockout:
            self._vetoes += 1
            return "bounded_surprise_lockout", False
        if self.terminal_calibrator.is_observed_danger(
            chosen.first_action,
            regime_index=self._regime_index,
        ):
            self._vetoes += 1
            return "bounded_observed_danger_veto", False
        raw_risk = _first_action_risk(chosen)
        marginal_risk = self.terminal_calibrator.calibrate(
            chosen.first_action,
            raw_risk,
            regime_index=self._regime_index,
        )
        if marginal_risk > self.config.bounded_maximum_terminal_risk + 1e-9:
            self._vetoes += 1
            return "bounded_marginal_risk_veto", False
        if context in self._observed_contexts:
            return "bounded_known_context", False
        if context in self._intervened_contexts:
            return "bounded_context_budget", False
        if self._interventions_this_reset >= self.config.bounded_maximum_interventions_per_reset:
            return "bounded_reset_budget", False
        self._intervened_contexts.add(context)
        self._interventions_this_reset += 1
        self._bounded_by_branch[self._branch_index] += 1
        return "bounded_override", True

    def observe_transition(self, record: Any) -> None:
        super().observe_transition(record)
        diagnostics = getattr(self.posterior, "last_update_diagnostics", None)
        surprise = getattr(diagnostics, "raw_mixture_surprise", None)
        if surprise is not None and float(surprise) > self.strong_surprise_threshold:
            self._surprise_lockout = True
            self._strong_surprises += 1

    def start_branch(self, *, regime_index: int | None = None) -> None:
        self._surprise_lockout = False
        super().start_branch(regime_index=regime_index)

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["bounded_safety"] = {
            "interventions_by_branch": dict(self._bounded_by_branch),
            "maximum_interventions_in_branch": max(self._bounded_by_branch.values(), default=0),
            "strong_surprises": self._strong_surprises,
            "surprise_lockout": self._surprise_lockout,
        }
        return payload


def build_controller(manifest: Mapping[str, Any]) -> BoundedTrajectoryController:
    caps = manifest["controller_caps"]
    executor = StructuralRoleProgramExecutor()
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    policy = T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2()
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=policy,
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
    return BoundedTrajectoryController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
            active_gate_passed=False,
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


def _factory(manifest: Mapping[str, Any], registry: dict[str, BoundedTrajectoryController]):
    def factory(game_id: str) -> UnifiedCognitiveController:
        sage_t = build_controller(manifest)
        registry[str(game_id)] = sage_t
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(
                sage_t_authority_mode="bounded",
                sage_t_counterfactual_gate_passed=True,
                sage_t_bounded_interventions_per_reset=int(
                    manifest["authority"]["maximum_interventions_per_reset"]
                ),
                sage_t_bounded_maximum_terminal_risk=float(
                    manifest["authority"]["maximum_marginal_terminal_risk"]
                ),
            ),
            sage_t_controller=sage_t,
        )

    return factory


def _off_factory(game_id: str) -> UnifiedCognitiveController:
    return UnifiedCognitiveController(
        game_id,
        config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
    )


def _flat_steps(arm: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for attempt in arm.get("attempts", ()) or ()
        for step in attempt.get("trace", ()) or ()
    ]


def _arm_metrics(arm: Mapping[str, Any]) -> dict[str, Any]:
    steps = _flat_steps(arm)
    progress = [
        max(0, int(step.get("levels_after", 0)) - int(step.get("levels_before", 0)))
        for step in steps
    ]
    terminals = [
        str(step.get("game_state_after", "")).upper()
        in live_base.TERMINAL_FAILURE_STATES
        for step in steps
    ]
    first = next((index for index, value in enumerate(progress) if value > 0), None)
    return {
        "actions": len(steps),
        "progress_events": sum(value > 0 for value in progress),
        "levels_completed": sum(progress),
        "game_over_actions": sum(terminals),
        "time_to_first_progress": first,
    }


def _intervention_metrics(
    bounded: Mapping[str, Any],
    controller: BoundedTrajectoryController,
) -> dict[str, int]:
    steps = _flat_steps(bounded)
    decisions = [
        record
        for record in controller.compact_records
        if record.get("kind") == "decision"
    ]
    applied = [bool(record.get("applied")) for record in decisions[: len(steps)]]
    useful = 0
    for index, value in enumerate(applied):
        if not value:
            continue
        window = steps[index : index + 3]
        useful += int(
            any(
                int(step.get("levels_after", 0)) > int(step.get("levels_before", 0))
                for step in window
            )
        )
    return {
        "interventions": sum(applied),
        "useful_interventions": useful,
        "wasted_interventions": sum(applied) - useful,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.inf
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


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
    rows = []
    registry: dict[str, BoundedTrajectoryController] = {}
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
            off = live_base._run_arm(controller_factory=_off_factory, **common)
            bounded = live_base._run_arm(
                controller_factory=_factory(manifest, registry),
                **common,
            )
            controller = registry[str(game_id)]
            rows.append(
                {
                    "game": str(game_id),
                    "seed": int(seed),
                    "off": _arm_metrics(off),
                    "bounded": _arm_metrics(bounded),
                    "intervention": _intervention_metrics(bounded, controller),
                    "controller_errors": len(tuple(bounded.get("controller_errors", ()) or ())),
                    "environment_errors": sum(
                        str(attempt.get("failure_cause", "")).startswith("environment_")
                        for attempt in bounded.get("attempts", ()) or ()
                    ),
                    "illegal_actions": sum(
                        "unavailable_decision" in str(error)
                        for error in bounded.get("controller_errors", ()) or ()
                    ),
                    "controller_summary": controller.summary(),
                }
            )
    elapsed = time.perf_counter() - started
    bounded_actions = sum(row["bounded"]["actions"] for row in rows)
    progress_events = sum(row["bounded"]["progress_events"] for row in rows)
    levels = sum(row["bounded"]["levels_completed"] for row in rows)
    game_over_delta = sum(
        row["bounded"]["game_over_actions"] - row["off"]["game_over_actions"]
        for row in rows
    )
    decisions = [
        value
        for controller in registry.values()
        for value in controller.decision_latencies_ms
    ]
    observations = [
        value
        for controller in registry.values()
        for value in controller.observation_latencies_ms
    ]
    false_high = []
    for controller in registry.values():
        records = controller.compact_records
        decision_records = [item for item in records if item.get("kind") == "decision"]
        observation_records = [item for item in records if item.get("kind") == "observation"]
        for decision, observation in zip(decision_records, observation_records):
            sequences = tuple(decision.get("sequences", ()) or ())
            actual_terminal = any(
                event == "game_over" for event in observation.get("events", ()) or ()
            )
            if sequences:
                false_high.append(
                    not actual_terminal
                    and float(sequences[0].get("terminal_risk", 0.0) or 0.0) >= 0.8
                )
    gate = manifest["gate"]
    checks = {
        "minimum_actions": bounded_actions >= int(gate["minimum_total_actions"]),
        "real_progress": progress_events >= int(gate["minimum_real_progress_events"]),
        "level_completed": levels >= int(gate["minimum_levels_completed"]),
        "game_over_nonincrease": game_over_delta <= int(gate["maximum_game_over_delta"]),
        "false_high_terminal": (
            mean(false_high) if false_high else 0.0
        ) <= float(gate["maximum_false_high_terminal_rate"]),
        "decision_p95": _quantile(decisions, 0.95) <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": _quantile(observations, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": elapsed <= float(gate["maximum_wall_seconds"]),
        "zero_illegal_actions": sum(row["illegal_actions"] for row in rows)
        <= int(gate["maximum_illegal_actions"]),
        "zero_controller_errors": sum(row["controller_errors"] for row in rows)
        <= int(gate["maximum_controller_errors"]),
        "zero_environment_errors": sum(row["environment_errors"] for row in rows)
        <= int(gate["maximum_environment_errors"]),
        "bounded_budget": all(
            int(row["controller_summary"]["bounded_safety"]["maximum_interventions_in_branch"])
            <= int(manifest["authority"]["maximum_interventions_per_reset"])
            for row in rows
        ),
        "effective_bounded_mode": all(
            row["controller_summary"]["effective_mode"] == "bounded" for row in rows
        ),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T9_3_PASSED" if passed else "T9_3_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": runtime,
        "checks": checks,
        "metrics": {
            "bounded_actions": bounded_actions,
            "progress_events": progress_events,
            "levels_completed": levels,
            "game_over_delta": game_over_delta,
            "false_high_terminal_rate": mean(false_high) if false_high else 0.0,
            "decision_p95_ms": _quantile(decisions, 0.95),
            "observation_p95_ms": _quantile(observations, 0.95),
            "wall_seconds": elapsed,
            "interventions": sum(row["intervention"]["interventions"] for row in rows),
            "useful_interventions": sum(
                row["intervention"]["useful_interventions"] for row in rows
            ),
            "wasted_interventions": sum(
                row["intervention"]["wasted_interventions"] for row in rows
            ),
        },
        "conditions": rows,
        "t9_4_authorized": passed,
        "active_authority_authorized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "conditions.jsonl", rows)
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
    return 0 if args.freeze or result.get("status") == "T9_3_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS_PER_RESET",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "RESETS",
    "SEED",
    "SOURCE_GAMES",
    "BoundedTrajectoryController",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_pilot",
]
