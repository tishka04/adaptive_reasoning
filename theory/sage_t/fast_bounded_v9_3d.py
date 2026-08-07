"""SAGE.T9.3d fast bounded confirmation on real source-train games."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
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
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import compact_bounded_v9_3c as compact
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v7 as live_i
from .controller import SageTConfig
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.3d-fast-bounded-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_3d_fast_bounded_manifest.json"
)
DEFAULT_PARTIAL_PATH = (
    Path("training")
    / "sage_t"
    / "bounded_v9_3c"
    / "conditions.partial.jsonl"
)
DEFAULT_PARTIAL_REPORT = (
    Path("training") / "sage_t" / "bounded_v9_3c" / "partial_report.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "bounded_v9_3d"
FAST_CAPS: Mapping[str, int] = {
    "maximum_programs": 32,
    "maximum_sequences": 8,
    "maximum_particles_per_decision": 4,
    "ordinary_horizon": 3,
    "maximum_structural_macros": 8,
    "maximum_executor_cache_entries": 256,
}


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "fast_bounded_v9_3d.py": v86c._file_sha256(
            directory / "fast_bounded_v9_3d.py"
        )
    }


def record_t9_3c_partial(
    *,
    partial_path: str | Path = DEFAULT_PARTIAL_PATH,
    output_path: str | Path = DEFAULT_PARTIAL_REPORT,
) -> dict[str, Any]:
    source = Path(partial_path)
    conditions = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(conditions) != 1 or conditions[0].get("game") != "lp85-305b61c3":
        raise ValueError("T9.3c partial checkpoint must contain exactly lp85")
    condition = conditions[0]
    bounded = condition.get("bounded", {})
    off = condition.get("off", {})
    if (
        int(bounded.get("levels_completed", 0)) < 1
        or int(off.get("levels_completed", 0)) != 0
        or int(bounded.get("game_over_actions", 0))
        > int(off.get("game_over_actions", 0))
    ):
        raise ValueError("T9.3c partial checkpoint lacks safe real progress")
    payload: dict[str, Any] = {
        "format_version": "sage-t9.3c-partial-result-v1",
        "status": "T9_3C_PARTIAL_BEHAVIORAL_PASS_RESOURCE_FAIL",
        "manifest_checksum": compact.load_manifest()["manifest_checksum"],
        "conditions": conditions,
        "complete_gate_result": False,
        "behavioral_evidence": {
            "bounded_levels": int(bounded["levels_completed"]),
            "baseline_levels": int(off["levels_completed"]),
            "game_over_delta": int(bounded["game_over_actions"])
            - int(off["game_over_actions"]),
        },
        "t9_4_authorized": False,
    }
    payload["report_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def _load_partial_report(
    path: str | Path = DEFAULT_PARTIAL_REPORT,
) -> dict[str, Any]:
    source = Path(path)
    report = (
        json.loads(source.read_text(encoding="utf-8"))
        if source.exists()
        else record_t9_3c_partial(output_path=source)
    )
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3c partial report checksum mismatch")
    if report.get("status") != "T9_3C_PARTIAL_BEHAVIORAL_PASS_RESOURCE_FAIL":
        raise ValueError("T9.3d requires the T9.3c partial result")
    if report.get("complete_gate_result") is not False:
        raise ValueError("T9.3c partial result cannot authorize T9.4")
    return report


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent = compact.load_manifest()
    partial = _load_partial_report()
    audit = compact._compact_prefix_audit(FAST_CAPS)
    if not audit["passed"]:
        raise ValueError("fast beam failed the frozen winning-prefix audit")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_3D_FAST_BOUNDED_SOURCE_TRAIN",
        "frozen_at": "2026-08-06",
        "parent_t9_3c_manifest_checksum": parent["manifest_checksum"],
        "parent_t9_3c_partial_checksum": partial["report_checksum"],
        "code_sha256": _code_hashes(),
        "controller_caps": dict(FAST_CAPS),
        "fast_prefix_audit": {
            key: value for key, value in audit.items() if key != "rows"
        },
        "selected_terminal_policy": parent["selected_terminal_policy"],
        "source_train_games": list(parent["source_train_games"]),
        "seeds": list(parent["seeds"]),
        "resets": int(parent["resets"]),
        "action_budget_per_reset": int(parent["action_budget_per_reset"]),
        "runtime": dict(parent["runtime"]),
        "authority": dict(parent["authority"]),
        "registered_changes": [
            "sequence beam 16 to 8 after 9/9 winning-prefix audit",
            "decision particles 8 to 4 after 9/9 winning-prefix audit",
            "executor cache 512 to 256 entries",
        ],
        "gate": {
            **dict(parent["gate"]),
            "maximum_peak_executor_cache_entries": 256,
        },
        "firewall": dict(parent["firewall"]),
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    v86c._write_json(
        Path(output_path).with_name("sage_t9_3d_fast_prefix_audit.json"),
        audit,
    )
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3d manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.3d manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_3D_FAST_BOUNDED_SOURCE_TRAIN":
        raise ValueError("T9.3d manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.3d code drifted")
    if payload.get("controller_caps") != dict(FAST_CAPS):
        raise ValueError("T9.3d caps drifted")
    if not bool(payload.get("fast_prefix_audit", {}).get("passed")):
        raise ValueError("T9.3d fast-prefix audit did not pass")
    if payload.get("parent_t9_3c_manifest_checksum") != compact.load_manifest().get(
        "manifest_checksum"
    ):
        raise ValueError("T9.3c manifest drifted")
    if payload.get("parent_t9_3c_partial_checksum") != _load_partial_report().get(
        "report_checksum"
    ):
        raise ValueError("T9.3c partial result drifted")
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
        raise ValueError("T9.3d firewall is open")
    return payload


class FastBoundedController(compact.CompactBoundedController):
    """Same bounded policy with the frozen fast counterfactual budget."""


def build_controller(manifest: Mapping[str, Any]) -> FastBoundedController:
    caps = manifest["controller_caps"]
    executor = StructuralRoleProgramExecutor(
        maximum_cache_entries=int(caps["maximum_executor_cache_entries"])
    )
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2(),
        maximum_particles=int(posterior_config["maximum_particles"]),
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
        maximum_repair_contexts=repair_r3.MAXIMUM_REPAIR_CONTEXTS,
    )
    authority = manifest["authority"]
    return FastBoundedController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(
                caps["maximum_particles_per_decision"]
            ),
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


def _factory(
    manifest: Mapping[str, Any],
    registry: dict[str, FastBoundedController],
):
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


def _run_condition(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    seed: int,
    environments_dir: str | Path,
    registry: dict[str, FastBoundedController],
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
    bounded = live_base._run_arm(
        controller_factory=_factory(manifest, registry),
        **common,
    )
    controller = registry[game_id]
    return {
        "game": game_id,
        "seed": seed,
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
        "compact_runtime": controller.summary()["compact_runtime"],
        "terminal_calibration": controller.terminal_calibrator.snapshot(),
    }


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
    registry: dict[str, FastBoundedController] = {}
    for game_id in manifest["source_train_games"]:
        for seed in manifest["seeds"]:
            conditions.append(
                _run_condition(
                    manifest=manifest,
                    game_id=str(game_id),
                    seed=int(seed),
                    environments_dir=environments_dir,
                    registry=registry,
                )
            )
            v86c._write_jsonl(destination / "conditions.partial.jsonl", conditions)

    elapsed = time.perf_counter() - started
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
    peak_cache = max(
        (
            int(row["compact_runtime"]["peak_executor_cache_entries"])
            for row in conditions
        ),
        default=0,
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
        "executor_cache_bounded": peak_cache
        <= int(gate["maximum_peak_executor_cache_entries"]),
        "zero_illegal_actions": sum(row["illegal_actions"] for row in conditions) == 0,
        "zero_controller_errors": sum(row["controller_errors"] for row in conditions) == 0,
        "zero_environment_errors": sum(row["environment_errors"] for row in conditions) == 0,
        "bounded_budget": all(
            int(row["bounded_safety"]["maximum_interventions_in_branch"])
            <= int(manifest["authority"]["maximum_interventions_per_reset"])
            for row in conditions
        ),
        "fast_prefix_audit": bool(manifest["fast_prefix_audit"]["passed"]),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report = {
        "format_version": FORMAT_VERSION,
        "status": "T9_3D_PASSED" if passed else "T9_3D_FAILED_CLOSED",
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
            "peak_executor_cache_entries": peak_cache,
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
    return 0 if args.freeze or result.get("status") == "T9_3D_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FAST_CAPS",
    "FORMAT_VERSION",
    "FastBoundedController",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "record_t9_3c_partial",
    "run_pilot",
]
