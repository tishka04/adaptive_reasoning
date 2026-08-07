"""SAGE.T9.3c compact bounded pilot after the T9.3b resource abort.

The policy change is deliberately narrow: preserve the live-grounded T9.3b
ordering, halve the sequence beam after proving the winning prefixes remain in
it, and clear the pure executor cache after every observation.  The latter is
semantics preserving and prevents a long live run from retaining thousands of
large counterfactual states.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import bounded_active_v9_3 as r1
from . import bounded_active_v9_3b as r2
from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v7 as live_i
from . import reachability_audit_v9 as t9_0
from . import trajectory_planning_v9_2 as t9_2
from .controller import SageTConfig
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler
from .terminal_calibration_v9 import T9_1_POLICIES

FORMAT_VERSION = "sage-t9.3c-compact-bounded-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_3c_compact_bounded_manifest.json"
)
DEFAULT_PARENT_ABORT = (
    Path("training") / "sage_t" / "bounded_v9_3b" / "abort.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "bounded_v9_3c"
COMPACT_CAPS: Mapping[str, int] = {
    "maximum_programs": 32,
    "maximum_sequences": 16,
    "maximum_particles_per_decision": 8,
    "ordinary_horizon": 3,
    "maximum_structural_macros": 8,
    "maximum_executor_cache_entries": 512,
}


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "compact_bounded_v9_3c.py": v86c._file_sha256(
            directory / "compact_bounded_v9_3c.py"
        )
    }


def record_parent_abort(
    *, output_path: str | Path = DEFAULT_PARENT_ABORT
) -> dict[str, Any]:
    """Record the resource abort without drawing behavioral conclusions."""

    parent_manifest = r2.load_manifest()
    payload: dict[str, Any] = {
        "format_version": "sage-t9.3b-resource-abort-v1",
        "status": "T9_3B_ABORTED_RESOURCE_LIMIT",
        "manifest_checksum": parent_manifest["manifest_checksum"],
        "reason": "private memory exceeded safe pilot budget before report emission",
        "observed_lower_bounds": {
            "cpu_seconds": 1220.0,
            "private_memory_bytes": 9_315_655_680,
        },
        "behavioral_result_available": False,
        "t9_4_authorized": False,
    }
    payload["report_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def _load_parent_abort(path: str | Path = DEFAULT_PARENT_ABORT) -> dict[str, Any]:
    source = Path(path)
    report = (
        json.loads(source.read_text(encoding="utf-8"))
        if source.exists()
        else record_parent_abort(output_path=source)
    )
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3b abort checksum mismatch")
    if report.get("status") != "T9_3B_ABORTED_RESOURCE_LIMIT":
        raise ValueError("T9.3c requires the registered T9.3b resource abort")
    if report.get("behavioral_result_available") is not False:
        raise ValueError("T9.3b abort must not claim a behavioral result")
    if report.get("t9_4_authorized") is not False:
        raise ValueError("T9.4 opened after an aborted T9.3b run")
    return report


def _compact_prefix_audit(caps: Mapping[str, int]) -> dict[str, Any]:
    manifest = t9_2.load_manifest()
    pairs = load_pairs(t9_0.DEFAULT_SHARD_DIR, t9_0.SOURCE_GAMES)
    paths = t9_0.winner_paths(pairs)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    rows = tuple(
        row
        for root_key, winning_paths in sorted(paths.items())
        for row in t9_2._audit_root(
            grouped[root_key],
            winning_path=winning_paths[0],
            condition="compact_h3",
            caps=caps,
            manifest=manifest,
        )
    )
    risks = [
        float(row.exact_first_action_risk)
        for row in rows
        if row.exact_first_action_risk is not None
    ]
    result = {
        "prefixes": len(rows),
        "exact_sequence_generated": sum(row.exact_sequence_generated for row in rows),
        "exact_sequence_top8": sum(
            row.exact_sequence_rank is not None and row.exact_sequence_rank <= 8
            for row in rows
        ),
        "correct_first_action": sum(row.selected_first_action for row in rows),
        "maximum_first_action_risk": max(risks, default=None),
        "decision_p95_ms": r1._quantile(
            [row.decision_latency_ms for row in rows], 0.95
        ),
        "execution_errors": sum(row.execution_errors for row in rows),
        "rows": [asdict(row) for row in rows],
    }
    result["passed"] = bool(
        result["prefixes"] == 9
        and result["exact_sequence_generated"] == 9
        and result["exact_sequence_top8"] == 9
        and result["correct_first_action"] == 9
        and result["maximum_first_action_risk"] is not None
        and float(result["maximum_first_action_risk"]) <= 0.05 + 1e-12
        and result["execution_errors"] == 0
    )
    return result


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent = _load_parent_abort()
    base = r2.load_manifest()
    audit = _compact_prefix_audit(COMPACT_CAPS)
    if not audit["passed"]:
        raise ValueError("compact beam failed the frozen winning-prefix audit")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_3C_COMPACT_BOUNDED_SOURCE_TRAIN",
        "frozen_at": "2026-08-06",
        "parent_t9_3b_manifest_checksum": base["manifest_checksum"],
        "parent_t9_3b_abort_checksum": parent["report_checksum"],
        "code_sha256": _code_hashes(),
        "controller_caps": dict(COMPACT_CAPS),
        "compact_prefix_audit": {
            key: value for key, value in audit.items() if key != "rows"
        },
        "selected_terminal_policy": base["selected_terminal_policy"],
        "source_train_games": list(base["source_train_games"]),
        "seeds": list(base["seeds"]),
        "resets": 3,
        "action_budget_per_reset": 20,
        "runtime": dict(base["runtime"]),
        "authority": dict(base["authority"]),
        "registered_changes": [
            "sequence beam 32 to 16 after exact winning-prefix equivalence",
            "executor cache 16384 to 512 entries",
            "pure executor cache cleared after observation and branch start",
            "20 actions per reset while retaining 120 bounded actions",
        ],
        "gate": {
            **dict(base["gate"]),
            "minimum_total_actions": 120,
            "maximum_wall_seconds": 360.0,
            "maximum_peak_executor_cache_entries": 512,
        },
        "firewall": dict(base["firewall"]),
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    v86c._write_json(
        Path(output_path).with_name("sage_t9_3c_compact_prefix_audit.json"),
        audit,
    )
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.3c manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.3c manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_3C_COMPACT_BOUNDED_SOURCE_TRAIN":
        raise ValueError("T9.3c manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.3c code drifted")
    if payload.get("controller_caps") != dict(COMPACT_CAPS):
        raise ValueError("T9.3c caps drifted")
    if not bool(payload.get("compact_prefix_audit", {}).get("passed")):
        raise ValueError("T9.3c compact-prefix audit did not pass")
    if payload.get("parent_t9_3b_manifest_checksum") != r2.load_manifest().get(
        "manifest_checksum"
    ):
        raise ValueError("T9.3b manifest drifted")
    if payload.get("parent_t9_3b_abort_checksum") != _load_parent_abort().get(
        "report_checksum"
    ):
        raise ValueError("T9.3b abort record drifted")
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
        raise ValueError("T9.3c firewall is open")
    return payload


class CompactBoundedController(r2.LiveGroundedBoundedController):
    """T9.3b behavior with a branch-local, bounded pure-execution cache."""

    peak_executor_cache_entries: int = 0

    def _capture_cache_peak(self) -> None:
        entries = int(self.executor.summary().get("cache_entries", 0))
        self.peak_executor_cache_entries = max(
            self.peak_executor_cache_entries,
            entries,
        )

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        result = super().decide(**kwargs)
        self._capture_cache_peak()
        return result

    def observe_transition(self, record: Any) -> None:
        super().observe_transition(record)
        self._capture_cache_peak()
        self.executor.clear_cache()

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self.executor.clear_cache()

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["compact_runtime"] = {
            "peak_executor_cache_entries": self.peak_executor_cache_entries,
            "current_executor_cache_entries": self.executor.summary()["cache_entries"],
        }
        return payload


def build_controller(manifest: Mapping[str, Any]) -> CompactBoundedController:
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
    return CompactBoundedController(
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
    registry: dict[str, CompactBoundedController],
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
    registry: dict[str, CompactBoundedController] = {}
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
            condition = {
                "game": str(game_id),
                "seed": int(seed),
                "off": r1._arm_metrics(off),
                "bounded": r1._arm_metrics(bounded),
                "intervention": r1._intervention_metrics(bounded, controller),
                "controller_errors": len(
                    tuple(bounded.get("controller_errors", ()) or ())
                ),
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
            conditions.append(condition)
            # Atomic partial result: a timeout never erases completed conditions.
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
        "compact_prefix_audit": bool(manifest["compact_prefix_audit"]["passed"]),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report = {
        "format_version": FORMAT_VERSION,
        "status": "T9_3C_PASSED" if passed else "T9_3C_FAILED_CLOSED",
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
    return 0 if args.freeze or result.get("status") == "T9_3C_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPACT_CAPS",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "CompactBoundedController",
    "build_controller",
    "freeze_manifest",
    "load_manifest",
    "main",
    "record_parent_abort",
    "run_pilot",
]
