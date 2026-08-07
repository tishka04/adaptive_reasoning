"""SAGE.T9.4 paired source-train active-authority gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import bounded_active_v9_3 as r1
from . import bounded_active_v9_3b as r2
from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v7 as live_i
from . import no_repair_bounded_v9_3f as bounded
from . import trajectory_planning_v9_2 as t9_2
from .controller import SageTConfig
from .decision import SequenceAssessment
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.4-paired-active-gate-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_4_paired_active_manifest.json"
)
DEFAULT_PARENT_REPORT = bounded.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "active_v9_4"
BOOTSTRAP_SEED = 9404
BOOTSTRAPS = 10_000


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "paired_active_gate_v9_4.py": v86c._file_sha256(
            directory / "paired_active_gate_v9_4.py"
        )
    }


def _load_parent_report(
    path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3f report checksum mismatch")
    if report.get("status") != "T9_3F_PASSED":
        raise ValueError("T9.3f did not pass")
    if report.get("t9_4_authorized") is not True:
        raise ValueError("T9.3f did not authorize T9.4")
    if report.get("active_authority_authorized") is not False:
        raise ValueError("active authority opened before T9.4")
    if not all(bool(value) for value in report.get("checks", {}).values()):
        raise ValueError("T9.3f contains a failed check")
    return report


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent_manifest = bounded.load_manifest()
    parent_report = _load_parent_report()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_4_PAIRED_ACTIVE_SOURCE_TRAIN",
        "frozen_at": "2026-08-06",
        "parent_t9_3f_manifest_checksum": parent_manifest["manifest_checksum"],
        "parent_t9_3f_report_checksum": parent_report["report_checksum"],
        "code_sha256": _code_hashes(),
        "controller_caps": dict(parent_manifest["controller_caps"]),
        "winning_prefix_audit": dict(parent_manifest["winning_prefix_audit"]),
        "selected_terminal_policy": parent_manifest["selected_terminal_policy"],
        "source_train_games": list(parent_manifest["source_train_games"]),
        "seeds": [0, 1, 2],
        "resets": 1,
        "action_budget_per_reset": 20,
        "runtime": dict(parent_manifest["runtime"]),
        "experimental_authority": {
            "mode": "active",
            "active_gate_passed_for_this_paired_experiment": True,
            "maximum_marginal_terminal_risk": 0.05,
            "observed_danger_is_absolute": True,
            "protected_route_is_absolute": True,
            "strong_surprise_lockout_threshold": 8.0,
            "repair_contexts": 0,
        },
        "pairing": {
            "same_game_seed_reset_and_action_budget": True,
            "bootstrap_unit": "game_seed",
            "bootstrap_samples": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gate": {
            "minimum_total_active_actions": 120,
            "minimum_total_level_advantage": 1,
            "minimum_nonnegative_games": 2,
            "strictly_positive_rate_delta_ci_lower": True,
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
            "production_active_authority": False,
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
        raise ValueError("T9.4 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.4 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_4_PAIRED_ACTIVE_SOURCE_TRAIN":
        raise ValueError("T9.4 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.4 code drifted")
    parent_manifest = bounded.load_manifest()
    if payload.get("parent_t9_3f_manifest_checksum") != parent_manifest.get(
        "manifest_checksum"
    ):
        raise ValueError("T9.3f manifest drifted")
    if payload.get("parent_t9_3f_report_checksum") != _load_parent_report().get(
        "report_checksum"
    ):
        raise ValueError("T9.3f report drifted")
    if payload.get("controller_caps") != parent_manifest.get("controller_caps"):
        raise ValueError("T9.4 controller caps drifted")
    if not bool(payload.get("winning_prefix_audit", {}).get("passed")):
        raise ValueError("T9.4 winning-prefix audit did not pass")
    firewall = payload.get("firewall", {})
    if any(
        bool(firewall.get(key))
        for key in (
            "source_validation_opened",
            "ar25_opened",
            "holdout_opened",
            "production_active_authority",
        )
    ):
        raise ValueError("T9.4 firewall is open")
    return payload


@dataclass
class SafeActiveDecisionEngine(r2.LiveGroundedDecisionEngine):
    """Apply the bounded first-action risk veto under active authority."""

    maximum_marginal_terminal_risk: float = 0.05

    def assess(self, *args: Any, **kwargs: Any) -> SequenceAssessment:
        assessment = super().assess(*args, **kwargs)
        if assessment.veto or self.calibrator is None:
            return assessment
        state = kwargs.get("state")
        regime = int(getattr(state, "regime_index", 0))
        first_action = assessment.first_action
        if self.calibrator.is_observed_danger(
            first_action,
            regime_index=regime,
        ):
            return replace(
                assessment,
                utility=float("-inf"),
                veto="active_observed_danger",
            )
        raw_risk = t9_2._first_action_risk(assessment)
        marginal_risk = (
            1.0
            if raw_risk is None
            else self.calibrator.calibrate(
                first_action,
                float(raw_risk),
                regime_index=regime,
            )
        )
        if marginal_risk > self.maximum_marginal_terminal_risk + 1e-9:
            return replace(
                assessment,
                utility=float("-inf"),
                veto="active_marginal_risk",
            )
        return assessment


class SafeActiveController(bounded.NoRepairBoundedController):
    """Active SAGE.T with the proven bounded safety envelope retained."""

    def __init__(
        self,
        *args: Any,
        maximum_marginal_terminal_risk: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_marginal_terminal_risk = float(
            maximum_marginal_terminal_risk
        )
        self.decision_engine = SafeActiveDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
            calibrator=self.terminal_calibrator,
            repeat_bonus_per_extra_action=self.repeat_bonus_per_extra_action,
            maximum_marginal_terminal_risk=self.maximum_marginal_terminal_risk,
        )

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["active_safety"] = {
            "maximum_marginal_terminal_risk": self.maximum_marginal_terminal_risk,
            "observed_danger_actions": self.terminal_calibrator.snapshot()[
                "danger_actions"
            ],
        }
        return payload


def build_controller(manifest: Mapping[str, Any]) -> SafeActiveController:
    caps = manifest["controller_caps"]
    executor = StructuralRoleProgramExecutor(
        maximum_cache_entries=int(caps["maximum_executor_cache_entries"])
    )
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2(),
        maximum_particles=min(
            int(posterior_config["maximum_particles"]),
            int(caps["maximum_programs"]),
        ),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(
            posterior_config["unknown_coverage_penalty"]
        ),
        repair_ess_threshold=float(
            posterior_config["repair_ess_threshold"]
        ),
        repair_log_likelihood_threshold=float(
            posterior_config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=int(caps["maximum_repair_contexts"]),
    )
    authority = manifest["experimental_authority"]
    return SafeActiveController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="active",
            counterfactual_gate_passed=True,
            active_gate_passed=True,
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(
                caps["maximum_particles_per_decision"]
            ),
            ordinary_horizon=int(caps["ordinary_horizon"]),
            bounded_maximum_terminal_risk=float(
                authority["maximum_marginal_terminal_risk"]
            ),
        ),
        terminal_policy=T9_1_POLICIES[str(manifest["selected_terminal_policy"])],
        maximum_structural_macros=int(caps["maximum_structural_macros"]),
        repeat_bonus_per_extra_action=0.35,
        strong_surprise_threshold=float(authority["strong_surprise_lockout_threshold"]),
        maximum_marginal_terminal_risk=float(
            authority["maximum_marginal_terminal_risk"]
        ),
    )


def _factory(
    manifest: Mapping[str, Any],
    registry: dict[tuple[str, int], SafeActiveController],
    seed: int,
):
    def factory(game_id: str) -> UnifiedCognitiveController:
        sage_t = build_controller(manifest)
        registry[(str(game_id), int(seed))] = sage_t
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(
                sage_t_authority_mode="active",
                sage_t_counterfactual_gate_passed=True,
                sage_t_active_gate_passed=True,
            ),
            sage_t_controller=sage_t,
        )

    return factory


def _paired_interval(
    values: Sequence[float],
    *,
    samples: int = BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    observed = tuple(float(value) for value in values)
    if not observed:
        return {"n": 0, "mean": math.nan, "lower_95": math.nan, "upper_95": math.nan}
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(observed) for _ in observed)
        for _ in range(max(1, int(samples)))
    )
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return {
        "n": len(observed),
        "mean": mean(observed),
        "lower_95": lower,
        "upper_95": upper,
    }


def _condition(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    seed: int,
    environments_dir: str | Path,
    registry: dict[tuple[str, int], SafeActiveController],
) -> dict[str, Any]:
    common = {
        "arm": "unified",
        "game_id": game_id,
        "seed": seed,
        "action_budget_per_reset": int(manifest["action_budget_per_reset"]),
        "resets": int(manifest["resets"]),
        "env_dir": Path(environments_dir),
        "env_factory": None,
    }
    off = live_base._run_arm(controller_factory=r1._off_factory, **common)
    active = live_base._run_arm(
        controller_factory=_factory(manifest, registry, seed),
        **common,
    )
    controller = registry[(game_id, seed)]
    off_metrics = r1._arm_metrics(off)
    active_metrics = r1._arm_metrics(active)
    active_actions = max(1, int(active_metrics["actions"]))
    off_actions = max(1, int(off_metrics["actions"]))
    return {
        "game": game_id,
        "seed": seed,
        "off": off_metrics,
        "active": active_metrics,
        "level_rate_delta": (
            float(active_metrics["levels_completed"]) / active_actions
            - float(off_metrics["levels_completed"]) / off_actions
        ),
        "intervention": r1._intervention_metrics(active, controller),
        "controller_errors": len(tuple(active.get("controller_errors", ()) or ())),
        "environment_errors": sum(
            str(attempt.get("failure_cause", "")).startswith("environment_")
            for attempt in active.get("attempts", ()) or ()
        ),
        "illegal_actions": sum(
            "unavailable_decision" in str(error)
            for error in active.get("controller_errors", ()) or ()
        ),
        "effective_mode": controller.summary()["effective_mode"],
        "active_safety": controller.summary()["active_safety"],
        "compact_runtime": controller.summary()["compact_runtime"],
    }


def run_gate(
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
            "active_authority_authorized": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "report.json", report)
        return report

    started = time.perf_counter()
    conditions = []
    registry: dict[tuple[str, int], SafeActiveController] = {}
    for game_id in manifest["source_train_games"]:
        for seed in manifest["seeds"]:
            conditions.append(
                _condition(
                    manifest=manifest,
                    game_id=str(game_id),
                    seed=int(seed),
                    environments_dir=environments_dir,
                    registry=registry,
                )
            )
            v86c._write_jsonl(destination / "conditions.partial.jsonl", conditions)

    elapsed = time.perf_counter() - started
    rate_interval = _paired_interval(
        [float(row["level_rate_delta"]) for row in conditions],
        samples=int(manifest["pairing"]["bootstrap_samples"]),
        seed=int(manifest["pairing"]["bootstrap_seed"]),
    )
    decisions = [
        value for controller in registry.values() for value in controller.decision_latencies_ms
    ]
    observations = [
        value
        for controller in registry.values()
        for value in controller.observation_latencies_ms
    ]
    terminal_rows = tuple(
        row
        for controller in registry.values()
        for row in r2._actual_action_terminal_rows(controller)
    )
    false_high_rate = (
        mean(not actual and risk >= 0.8 for risk, actual in terminal_rows)
        if terminal_rows
        else 0.0
    )
    active_actions = sum(row["active"]["actions"] for row in conditions)
    active_levels = sum(row["active"]["levels_completed"] for row in conditions)
    off_levels = sum(row["off"]["levels_completed"] for row in conditions)
    game_over_delta = sum(
        row["active"]["game_over_actions"] - row["off"]["game_over_actions"]
        for row in conditions
    )
    by_game = {
        game: sum(
            row["active"]["levels_completed"] - row["off"]["levels_completed"]
            for row in conditions
            if row["game"] == game
        )
        for game in manifest["source_train_games"]
    }
    gate = manifest["gate"]
    checks = {
        "minimum_actions": active_actions >= int(gate["minimum_total_active_actions"]),
        "level_advantage": active_levels - off_levels
        >= int(gate["minimum_total_level_advantage"]),
        "paired_rate_ci_positive": float(rate_interval["lower_95"]) > 0.0,
        "nonnegative_games": sum(value >= 0 for value in by_game.values())
        >= int(gate["minimum_nonnegative_games"]),
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
        "effective_active_mode": all(
            row["effective_mode"] == "active" for row in conditions
        ),
        "safety_envelope": all(
            float(row["active_safety"]["maximum_marginal_terminal_risk"])
            <= 0.05
            for row in conditions
        ),
        "winning_prefix_audit": bool(manifest["winning_prefix_audit"]["passed"]),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report = {
        "format_version": FORMAT_VERSION,
        "status": "T9_4_PASSED" if passed else "T9_4_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": runtime,
        "checks": checks,
        "metrics": {
            "active_actions": active_actions,
            "active_levels_completed": active_levels,
            "baseline_levels_completed": off_levels,
            "total_level_advantage": active_levels - off_levels,
            "level_rate_delta_interval": rate_interval,
            "per_game_level_advantage": by_game,
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
        "active_authority_authorized": passed,
        "source_validation_active_gate_authorized": passed,
        "source_validation_opened": False,
        "holdout_opened": False,
        "ar25_opened": False,
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
        result = run_gate(
            manifest_path=args.manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_4_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAPS",
    "BOOTSTRAP_SEED",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "SafeActiveController",
    "SafeActiveDecisionEngine",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
