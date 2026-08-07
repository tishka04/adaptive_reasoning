"""SAGE.T9.6 source-train gate for branch-local productive abstention.

T9.6 is deliberately downstream of the failed, frozen T9.5 validation gate.
It changes no program, posterior, utility, or safety coefficient.  The only
challenger change is a branch-local authority budget: after five overrides
without a level transition, SAGE.T yields to the unchanged baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from . import bounded_active_v9_3 as r1
from . import bounded_active_v9_3b as r2
from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot as live_base
from . import paired_active_gate_v9_4 as active
from . import source_validation_active_v9_5 as validation

FORMAT_VERSION = "sage-t9.6-productive-abstention-gate-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_6_productive_abstention_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "productive_abstention_v9_6"
DEFAULT_T9_4_REPORT = active.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_T9_5_REPORT = validation.DEFAULT_OUTPUT_DIR / "report.json"
BOOTSTRAP_SEED = 9606
BOOTSTRAPS = 10_000


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "productive_abstention_v9_6.py": v86c._file_sha256(
            directory / "productive_abstention_v9_6.py"
        )
    }


def _load_signed_report(
    path: str | Path,
    *,
    expected_status: str,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError(f"report checksum mismatch: {path}")
    if report.get("status") != expected_status:
        raise ValueError(f"unexpected report status: {path}")
    return report


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent_manifest = active.load_manifest()
    parent_report = _load_signed_report(
        DEFAULT_T9_4_REPORT,
        expected_status="T9_4_PASSED",
    )
    validation_report = _load_signed_report(
        DEFAULT_T9_5_REPORT,
        expected_status="T9_5_FAILED_CLOSED",
    )
    reference = parent_report["metrics"]
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_6_SOURCE_TRAIN_ABSTENTION",
        "frozen_at": "2026-08-06",
        "parent_t9_4_manifest_checksum": parent_manifest["manifest_checksum"],
        "parent_t9_4_report_checksum": parent_report["report_checksum"],
        "parent_t9_5_report_checksum": validation_report["report_checksum"],
        "code_sha256": _code_hashes(),
        "controller_caps": dict(parent_manifest["controller_caps"]),
        "selected_terminal_policy": parent_manifest["selected_terminal_policy"],
        "source_train_games": list(parent_manifest["source_train_games"]),
        "seeds": list(parent_manifest["seeds"]),
        "resets": int(parent_manifest["resets"]),
        "action_budget_per_reset": int(parent_manifest["action_budget_per_reset"]),
        "experimental_authority": {
            **dict(parent_manifest["experimental_authority"]),
            "maximum_unproductive_interventions_per_branch": 5,
            "budget_resets_on_level_change": True,
            "fallback_after_budget": True,
            "policy_change": "authority_abstention_only",
        },
        "reference_t9_4": {
            "active_levels_completed": int(reference["active_levels_completed"]),
            "baseline_levels_completed": int(reference["baseline_levels_completed"]),
            "interventions": int(reference["interventions"]),
            "useful_interventions": int(reference["useful_interventions"]),
            "wasted_interventions": int(reference["wasted_interventions"]),
            "game_over_delta": int(reference["game_over_delta"]),
        },
        "pairing": {
            "same_game_seed_reset_and_action_budget_as_t9_4": True,
            "bootstrap_unit": "game_seed",
            "bootstrap_samples": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "gate": {
            "minimum_retained_levels": int(reference["active_levels_completed"]),
            "minimum_retained_useful_interventions": int(
                reference["useful_interventions"]
            ),
            "minimum_intervention_reduction_fraction": 0.5,
            "maximum_game_over_delta": int(reference["game_over_delta"]),
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
        raise ValueError("T9.6 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.6 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_6_SOURCE_TRAIN_ABSTENTION":
        raise ValueError("T9.6 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.6 code drifted")

    parent = active.load_manifest()
    parent_report = _load_signed_report(
        DEFAULT_T9_4_REPORT,
        expected_status="T9_4_PASSED",
    )
    validation_report = _load_signed_report(
        DEFAULT_T9_5_REPORT,
        expected_status="T9_5_FAILED_CLOSED",
    )
    if payload.get("parent_t9_4_manifest_checksum") != parent["manifest_checksum"]:
        raise ValueError("T9.4 manifest drifted")
    if payload.get("parent_t9_4_report_checksum") != parent_report["report_checksum"]:
        raise ValueError("T9.4 report drifted")
    if payload.get("parent_t9_5_report_checksum") != validation_report["report_checksum"]:
        raise ValueError("T9.5 report drifted")
    if payload.get("controller_caps") != parent["controller_caps"]:
        raise ValueError("T9.6 controller caps drifted")
    if payload.get("source_train_games") != parent["source_train_games"]:
        raise ValueError("T9.6 source-train games drifted")
    if payload.get("seeds") != parent["seeds"]:
        raise ValueError("T9.6 seeds drifted")
    authority = payload.get("experimental_authority", {})
    if int(authority.get("maximum_unproductive_interventions_per_branch", 0)) != 5:
        raise ValueError("T9.6 abstention budget drifted")
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
        raise ValueError("T9.6 firewall is open")
    return payload


class ProductiveAbstainingController(active.SafeActiveController):
    """Active controller that yields after a nonproductive branch budget."""

    def __init__(
        self,
        *args: Any,
        maximum_unproductive_interventions_per_branch: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_unproductive_interventions_per_branch = max(
            0,
            int(maximum_unproductive_interventions_per_branch),
        )
        self._productive_budget_used = 0
        self._productive_abstentions = 0
        self._productive_budget_resets = 0

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        if (
            self._productive_budget_used
            >= self.maximum_unproductive_interventions_per_branch
        ):
            started = time.perf_counter()
            try:
                self._productive_abstentions += 1
                return self._fallback(
                    str(kwargs.get("symbolic_action_name", "")).strip().upper(),
                    dict(kwargs.get("symbolic_action_data") or {}),
                    reason="active_unproductive_abstention",
                )
            finally:
                self.decision_latencies_ms.append(
                    (time.perf_counter() - started) * 1000.0
                )
        arbitration = super().decide(**kwargs)
        if arbitration.applied:
            self._productive_budget_used += 1
        return arbitration

    def start_branch(self, *, regime_index: int | None = None) -> None:
        self._productive_budget_used = 0
        self._productive_budget_resets += 1
        super().start_branch(regime_index=regime_index)

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["productive_abstention"] = {
            "maximum_unproductive_interventions_per_branch": (
                self.maximum_unproductive_interventions_per_branch
            ),
            "budget_used": self._productive_budget_used,
            "abstentions": self._productive_abstentions,
            "budget_resets": self._productive_budget_resets,
        }
        return payload


def build_controller(manifest: Mapping[str, Any]) -> ProductiveAbstainingController:
    caps = manifest["controller_caps"]
    executor = active.StructuralRoleProgramExecutor(
        maximum_cache_entries=int(caps["maximum_executor_cache_entries"])
    )
    t7 = active.v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    posterior = active.BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=active.T8_6G_POLICIES[
            active.live_i.SELECTED_POLICY
        ].with_repair_v2(),
        maximum_particles=min(
            int(posterior_config["maximum_particles"]),
            int(caps["maximum_programs"]),
        ),
        channel_weights=active.v86._weights("joint"),
        unknown_coverage_penalty=float(
            posterior_config["unknown_coverage_penalty"]
        ),
        repair_ess_threshold=float(posterior_config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            posterior_config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=int(caps["maximum_repair_contexts"]),
    )
    authority = manifest["experimental_authority"]
    return ProductiveAbstainingController(
        executor=executor,
        posterior=posterior,
        proposer=active.live_i.StructuralGoalFragmentProposer(),
        assembler=active.ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=active.SageTConfig(
            mode="active",
            counterfactual_gate_passed=True,
            active_gate_passed=True,
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(caps["maximum_particles_per_decision"]),
            ordinary_horizon=int(caps["ordinary_horizon"]),
            bounded_maximum_terminal_risk=float(
                authority["maximum_marginal_terminal_risk"]
            ),
        ),
        terminal_policy=active.T9_1_POLICIES[str(manifest["selected_terminal_policy"])],
        maximum_structural_macros=int(caps["maximum_structural_macros"]),
        repeat_bonus_per_extra_action=0.35,
        strong_surprise_threshold=float(authority["strong_surprise_lockout_threshold"]),
        maximum_marginal_terminal_risk=float(
            authority["maximum_marginal_terminal_risk"]
        ),
        maximum_unproductive_interventions_per_branch=int(
            authority["maximum_unproductive_interventions_per_branch"]
        ),
    )


def _factory(
    manifest: Mapping[str, Any],
    registry: dict[tuple[str, int], ProductiveAbstainingController],
    seed: int,
):
    def factory(game_id: str) -> active.UnifiedCognitiveController:
        sage_t = build_controller(manifest)
        registry[(str(game_id), int(seed))] = sage_t
        return active.UnifiedCognitiveController(
            game_id,
            config=active.UnifiedCognitiveConfig(
                sage_t_authority_mode="active",
                sage_t_counterfactual_gate_passed=True,
                sage_t_active_gate_passed=True,
            ),
            sage_t_controller=sage_t,
        )

    return factory


def _condition(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    seed: int,
    environments_dir: str | Path,
    registry: dict[tuple[str, int], ProductiveAbstainingController],
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
    selected = live_base._run_arm(
        controller_factory=_factory(manifest, registry, seed),
        **common,
    )
    controller = registry[(game_id, seed)]
    off_metrics = r1._arm_metrics(off)
    active_metrics = r1._arm_metrics(selected)
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
        "intervention": r1._intervention_metrics(selected, controller),
        "abstention": dict(controller.summary()["productive_abstention"]),
        "controller_errors": len(tuple(selected.get("controller_errors", ()) or ())),
        "environment_errors": sum(
            str(attempt.get("failure_cause", "")).startswith("environment_")
            for attempt in selected.get("attempts", ()) or ()
        ),
        "illegal_actions": sum(
            "unavailable_decision" in str(error)
            for error in selected.get("controller_errors", ()) or ()
        ),
        "effective_mode": controller.summary()["effective_mode"],
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
            "holdout_authorized": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "report.json", report)
        return report

    started = time.perf_counter()
    registry: dict[tuple[str, int], ProductiveAbstainingController] = {}
    conditions = []
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
    rate_interval = active._paired_interval(
        [float(row["level_rate_delta"]) for row in conditions],
        samples=int(manifest["pairing"]["bootstrap_samples"]),
        seed=int(manifest["pairing"]["bootstrap_seed"]),
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
    active_levels = sum(row["active"]["levels_completed"] for row in conditions)
    off_levels = sum(row["off"]["levels_completed"] for row in conditions)
    game_over_delta = sum(
        row["active"]["game_over_actions"] - row["off"]["game_over_actions"]
        for row in conditions
    )
    interventions = sum(row["intervention"]["interventions"] for row in conditions)
    useful = sum(row["intervention"]["useful_interventions"] for row in conditions)
    wasted = interventions - useful
    abstentions = sum(row["abstention"]["abstentions"] for row in conditions)
    reference = manifest["reference_t9_4"]
    reference_interventions = max(1, int(reference["interventions"]))
    intervention_reduction = 1.0 - interventions / reference_interventions
    gate = manifest["gate"]
    checks = {
        "levels_retained": active_levels >= int(gate["minimum_retained_levels"]),
        "useful_interventions_retained": useful
        >= int(gate["minimum_retained_useful_interventions"]),
        "interventions_reduced": intervention_reduction
        >= float(gate["minimum_intervention_reduction_fraction"]),
        "abstention_exercised": abstentions > 0,
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
        "effective_active_mode": all(row["effective_mode"] == "active" for row in conditions),
        "source_train_only": bool(manifest["firewall"]["source_train_only"]),
        "holdout_closed": manifest["firewall"]["holdout_opened"] is False,
    }
    passed = all(checks.values())
    report = {
        "format_version": FORMAT_VERSION,
        "status": "T9_6_PASSED" if passed else "T9_6_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": runtime,
        "checks": checks,
        "metrics": {
            "active_levels_completed": active_levels,
            "baseline_levels_completed": off_levels,
            "level_rate_delta_interval": rate_interval,
            "game_over_delta": game_over_delta,
            "false_high_terminal_rate": false_high_rate,
            "interventions": interventions,
            "useful_interventions": useful,
            "wasted_interventions": wasted,
            "abstentions": abstentions,
            "intervention_reduction_fraction": intervention_reduction,
            "decision_p95_ms": r1._quantile(decisions, 0.95),
            "observation_p95_ms": r1._quantile(observations, 0.95),
            "wall_seconds": elapsed,
        },
        "conditions": conditions,
        "source_validation_retest_authorized": passed,
        "holdout_authorized": False,
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
    return 0 if args.freeze or result.get("status") == "T9_6_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "ProductiveAbstainingController",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
