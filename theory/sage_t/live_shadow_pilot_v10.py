"""T8.6j-r3 extended live shadow confirmation with seven resets."""

from __future__ import annotations

import argparse
import json
import math
import time
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
from . import calibration_gate_v8_6j_r3 as r3
from . import live_shadow_pilot_v5 as t8_5
from . import live_shadow_pilot_v6 as t8_6_live
from . import live_shadow_pilot_v7 as live_i
from . import live_shadow_pilot_v9 as live_r2
from .controller import SageTConfig
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .structural_roles import StructuralRoleProgramExecutor
from .synthesis import ProgramAssembler

FORMAT_VERSION = "sage-t8.6j-r3-extended-live-v1"
DEFAULT_ACTION_MANIFEST = Path(__file__).with_name(
    "sage_t8_6j_r3_extended_action_manifest.json"
)
DEFAULT_CONFIRMATION_MANIFEST = Path(__file__).with_name(
    "sage_t8_6j_r3_extended_confirmation_manifest.json"
)
DEFAULT_R3_REPORT = r3.DEFAULT_OUTPUT_DIR / "gate_report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6j_r3_live"
SELECTED_POLICY = r3.SELECTED_POLICY
RESETS = 7
ACTIONS_PER_RESET = 50
MAXIMUM_ACTIONS_PER_GAME = RESETS * ACTIONS_PER_RESET
MAXIMUM_TOTAL_ACTIONS = 2 * MAXIMUM_ACTIONS_PER_GAME
MINIMUM_ACTIONS_PER_GAME = 200
MINIMUM_TOTAL_ACTIONS = 400


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "live_shadow_pilot_v10.py": v86c._file_sha256(
            directory / "live_shadow_pilot_v10.py"
        )
    }


def _load_r3_report(path: str | Path = DEFAULT_R3_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j-r3 gate report checksum mismatch")
    if report.get("status") != "READY_FOR_T8_6J_R3_LONG_LIVE":
        raise ValueError("T8.6j-r3 gate did not authorize long live")
    if not all(bool(value) for value in report.get("checks", {}).values()):
        raise ValueError("T8.6j-r3 report contains a failed gate")
    if report.get("source_validation_authorized") is not False:
        raise ValueError("source-validation opened before extended live")
    return report


def freeze_action_manifest(
    *,
    output_path: str | Path = DEFAULT_ACTION_MANIFEST,
    parent_path: str | Path = live_r2.DEFAULT_ACTION_MANIFEST,
) -> dict[str, Any]:
    parent = t8_5.load_frozen_manifest(parent_path)
    payload = json.loads(json.dumps(parent))
    payload.pop("manifest_checksum", None)
    payload["frozen_at"] = "2026-08-06"
    payload["action_budget_per_reset"] = ACTIONS_PER_RESET
    payload["resets"] = RESETS
    payload["challenger"] = {
        "label": "T8_6J_R3_BOUNDED_REPAIR_EXTENDED_HORIZON",
        "purpose": "obtain at least 200 realized actions per source-train game",
        "parent_action_manifest_checksum": parent["manifest_checksum"],
        "can_satisfy_t8_0_gate": False,
    }
    payload["gate"] = {
        **dict(parent["gate"]),
        "minimum_actions": MINIMUM_TOTAL_ACTIONS,
        "minimum_finite_surprise_samples": MINIMUM_TOTAL_ACTIONS,
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def freeze_confirmation_manifest(
    *,
    output_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    r3_manifest_path: str | Path = r3.DEFAULT_MANIFEST_PATH,
    r3_report_path: str | Path = DEFAULT_R3_REPORT,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
) -> dict[str, Any]:
    repair_gate = r3.load_manifest(r3_manifest_path)
    repair_report = _load_r3_report(r3_report_path)
    action_manifest = t8_5.load_frozen_manifest(action_manifest_path)
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_6J_R3_EXTENDED_LIVE",
        "frozen_at": "2026-08-06",
        "r3_manifest_checksum": repair_gate["manifest_checksum"],
        "r3_report_checksum": repair_report["report_checksum"],
        "action_manifest_checksum": action_manifest["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "selected_challenger": SELECTED_POLICY,
        "repair_policy": v86c._json_safe(policy.__dict__),
        "maximum_repair_contexts_per_game": r3.MAXIMUM_REPAIR_CONTEXTS,
        "source_train_games": list(action_manifest["source_train_games"]),
        "resets": RESETS,
        "actions_per_reset": ACTIONS_PER_RESET,
        "maximum_actions_per_game": MAXIMUM_ACTIONS_PER_GAME,
        "maximum_actions": MAXIMUM_TOTAL_ACTIONS,
        "minimum_actions_per_game": MINIMUM_ACTIONS_PER_GAME,
        "minimum_actions": MINIMUM_TOTAL_ACTIONS,
        "maximum_environment_interactions": MAXIMUM_TOTAL_ACTIONS * 2,
        "seeds": [0],
        "authority": "shadow",
        "gate": {
            "minimum_prediction_coverage": 1.0,
            "maximum_decision_p95_ms": 2500.0,
            "maximum_observation_p95_ms": 3000.0,
            "maximum_wall_seconds": 720.0,
            "latency_window": 20,
            "maximum_per_game_decision_tail_ratio": 2.0,
            "maximum_per_game_observation_tail_ratio": 2.0,
            "maximum_repairs_per_observation": 1,
            "maximum_surviving_children_per_observation": 4,
            "require_repair_budget_exercised": True,
            "require_incremental_cache_activity": True,
        },
        "source_validation_authorized": False,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_confirmation_manifest(
    path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    *,
    r3_manifest_path: str | Path = r3.DEFAULT_MANIFEST_PATH,
    r3_report_path: str | Path = DEFAULT_R3_REPORT,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j-r3 extended-live manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T8.6j-r3 extended-live manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6J_R3_EXTENDED_LIVE":
        raise ValueError("T8.6j-r3 extended-live manifest is not frozen")
    repair_gate = r3.load_manifest(r3_manifest_path)
    repair_report = _load_r3_report(r3_report_path)
    action_manifest = t8_5.load_frozen_manifest(action_manifest_path)
    if payload.get("r3_manifest_checksum") != repair_gate.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6j-r3 manifest drifted")
    if payload.get("r3_report_checksum") != repair_report.get(
        "report_checksum"
    ):
        raise ValueError("T8.6j-r3 report drifted")
    if payload.get("action_manifest_checksum") != action_manifest.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.6j-r3 action protocol drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T8.6j-r3 live code drifted")
    if payload.get("authority") != "shadow" or any(
        bool(payload.get(key))
        for key in (
            "source_validation_authorized",
            "bounded_authority_authorized",
            "active_authority_authorized",
        )
    ):
        raise ValueError("T8.6j-r3 live firewall is open")
    return payload, repair_report


def _controller(*, caps: Mapping[str, Any]) -> t8_5.MaterializedActionController:
    executor = StructuralRoleProgramExecutor()
    t7 = v86.load_t7_manifest(verify_code=True)
    policy = T8_6G_POLICIES[SELECTED_POLICY].with_repair_v2()
    config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=policy,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=r3.MAXIMUM_REPAIR_CONTEXTS,
    )
    return t8_5.MaterializedActionController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="shadow",
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(
                caps["maximum_particles_per_decision"]
            ),
            ordinary_horizon=int(caps["ordinary_horizon"]),
        ),
    )


class ExtendedLiveController(live_i.StructuralLiveController):
    def __init__(self, *, caps: Mapping[str, Any]) -> None:
        selected = _controller(caps=caps)
        self.selected_name = T8_6G_POLICIES[
            SELECTED_POLICY
        ].with_repair_v2().name
        self.controllers = {self.selected_name: selected}
        self.selected = selected
        self.posterior = selected.posterior


def _factory_builder(*, registry: dict[str, ExtendedLiveController]) -> Any:
    def builder(*, mode: str, manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
        caps = manifest["controller"]

        def factory(game_id: str) -> UnifiedCognitiveController:
            if mode == "off":
                return UnifiedCognitiveController(
                    game_id,
                    config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
                )
            sage_t = ExtendedLiveController(caps=caps)
            registry[str(game_id)] = sage_t
            return UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="shadow"),
                sage_t_controller=sage_t,  # type: ignore[arg-type]
            )

        return factory

    return builder


def _extended_checks(
    policy_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    *,
    selected_condition: str,
    base_report: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    registry: Mapping[str, ExtendedLiveController],
) -> dict[str, Any]:
    selected = [
        row for row in policy_rows if row["condition"] == selected_condition
    ]
    gate = confirmation["gate"]
    decisions = [
        float(row["decision_latency_ms"])
        for row in selected
        if row.get("decision_latency_ms") is not None
    ]
    observations = [
        float(row["observation_latency_ms"])
        for row in selected
        if row.get("observation_latency_ms") is not None
    ]
    by_game: dict[str, int] = {}
    for row in selected:
        by_game[str(row["game"])] = by_game.get(str(row["game"]), 0) + 1
    window = int(gate["latency_window"])
    decision_tails = live_r2._tail_ratios(
        selected, value_key="decision_latency_ms", window=window
    )
    observation_tails = live_r2._tail_ratios(
        selected, value_key="observation_latency_ms", window=window
    )
    repairs = live_r2._repair_metrics(base_rows, registry)  # type: ignore[arg-type]
    safety = dict(base_report.get("safety", {}))
    coverage = (
        mean(float(bool(row["prediction_available"])) for row in selected)
        if selected
        else 0.0
    )
    performance = repairs["performance"]
    budget_skips = sum(
        int(item.get("repair_budget_skips", 0))
        for item in performance.values()
    )
    budget_respected = all(
        int(item.get("unique_repair_contexts", 0))
        <= int(confirmation["maximum_repair_contexts_per_game"])
        for item in performance.values()
    )
    checks = {
        "minimum_total_actions": len(selected)
        >= int(confirmation["minimum_actions"]),
        "minimum_actions_per_game": len(by_game)
        == len(confirmation["source_train_games"])
        and min(by_game.values(), default=0)
        >= int(confirmation["minimum_actions_per_game"]),
        "prediction_coverage": coverage
        >= float(gate["minimum_prediction_coverage"]),
        "same_actions": bool(safety.get("actions_match")),
        "same_resets": bool(safety.get("same_reset_states")),
        "zero_interventions": int(safety.get("interventions", 0)) == 0,
        "zero_illegal_actions": int(safety.get("illegal_actions", 0)) == 0,
        "zero_controller_errors": int(safety.get("controller_errors", 0)) == 0,
        "zero_environment_errors": int(safety.get("environment_errors", 0)) == 0,
        "decision_p95": bool(decisions)
        and t8_6_live._quantile(decisions, 0.95)
        <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": bool(observations)
        and t8_6_live._quantile(observations, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": float(base_report.get("wall_clock_seconds", math.inf))
        <= float(gate["maximum_wall_seconds"]),
        "per_game_decision_tail": bool(decision_tails)
        and max(decision_tails.values())
        <= float(gate["maximum_per_game_decision_tail_ratio"]),
        "per_game_observation_tail": bool(observation_tails)
        and max(observation_tails.values())
        <= float(gate["maximum_per_game_observation_tail_ratio"]),
        "repair_budget": repairs["maximum_attempted_per_observation"]
        <= int(gate["maximum_repairs_per_observation"])
        and repairs["maximum_admitted_per_observation"]
        <= int(gate["maximum_surviving_children_per_observation"]),
        "global_repair_budget": budget_respected,
        "repair_budget_exercised": (
            not bool(gate["require_repair_budget_exercised"])
            or budget_skips > 0
        ),
        "incremental_cache_activity": (
            not bool(gate["require_incremental_cache_activity"])
            or repairs["semantic_cache_hits"] > 0
        ),
        "no_semantic_collapse": not any(
            bool(row["semantic_collapse"]) for row in selected
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "actions": len(selected),
            "actions_by_game": by_game,
            "environment_interactions": len(selected) * 2,
            "prediction_coverage": coverage,
            "decision_p95_ms": t8_6_live._quantile(decisions, 0.95),
            "observation_p95_ms": t8_6_live._quantile(observations, 0.95),
            "wall_clock_seconds": base_report.get("wall_clock_seconds"),
            "per_game_decision_tail_ratio": decision_tails,
            "per_game_observation_tail_ratio": observation_tails,
            "semantic_collapses": sum(
                bool(row["semantic_collapse"]) for row in selected
            ),
            "repair_budget_skips": budget_skips,
            "repairs": repairs,
        },
    }


def run_extended_live(
    *,
    confirmation_manifest_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    r3_manifest_path: str | Path = r3.DEFAULT_MANIFEST_PATH,
    r3_report_path: str | Path = DEFAULT_R3_REPORT,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    confirmation, repair_report = load_confirmation_manifest(
        confirmation_manifest_path,
        r3_manifest_path=r3_manifest_path,
        r3_report_path=r3_report_path,
        action_manifest_path=action_manifest_path,
    )
    registry: dict[str, ExtendedLiveController] = {}
    previous_factory = t8_5._controller_factory
    t8_5._controller_factory = _factory_builder(registry=registry)
    started = time.perf_counter()
    try:
        base_report = t8_5.run_live_shadow_pilot(
            manifest_path=action_manifest_path,
            environments_dir=environments_dir,
            output_dir=output_dir,
        )
    finally:
        t8_5._controller_factory = previous_factory
    destination = Path(output_dir)
    base_rows = [
        json.loads(line)
        for line in (destination / "rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    policy_rows = t8_6_live._policy_live_rows(registry, base_rows)
    selected_condition = str(confirmation["repair_policy"]["name"])
    live = _extended_checks(
        policy_rows,
        base_rows,
        selected_condition=selected_condition,
        base_report=base_report,
        confirmation=confirmation,
        registry=registry,
    )
    passed = bool(live["passed"])
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": (
            "READY_TO_PREPARE_T8_7_SOURCE_VALIDATION"
            if passed
            else "T8_6J_R3_EXTENDED_LIVE_FAILED_CLOSED"
        ),
        "manifest_checksum": confirmation["manifest_checksum"],
        "r3_report_checksum": repair_report["report_checksum"],
        "selected_challenger": confirmation["selected_challenger"],
        "live_confirmation": live,
        "base_live_report_checksum": base_report.get("report_checksum"),
        "elapsed_seconds": time.perf_counter() - started,
        "conclusion": (
            "CALIBRATION_RECOVERED" if passed else "INCONCLUSIVE_FAIL_CLOSED"
        ),
        "source_validation_authorized": passed,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "policy_rows.jsonl", policy_rows)
    v86c._write_json(destination / "t8_6j_r3_extended_live_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-manifest", default=str(DEFAULT_CONFIRMATION_MANIFEST)
    )
    parser.add_argument("--action-manifest", default=str(DEFAULT_ACTION_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--freeze-actions", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze_actions:
        result = freeze_action_manifest(output_path=args.action_manifest)
    elif args.freeze:
        result = freeze_confirmation_manifest(
            output_path=args.confirmation_manifest,
            action_manifest_path=args.action_manifest,
        )
    else:
        result = run_extended_live(
            confirmation_manifest_path=args.confirmation_manifest,
            action_manifest_path=args.action_manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze_actions or args.freeze or result.get(
        "source_validation_authorized"
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS_PER_RESET",
    "DEFAULT_ACTION_MANIFEST",
    "DEFAULT_CONFIRMATION_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "MAXIMUM_ACTIONS_PER_GAME",
    "MAXIMUM_TOTAL_ACTIONS",
    "MINIMUM_ACTIONS_PER_GAME",
    "MINIMUM_TOTAL_ACTIONS",
    "RESETS",
    "ExtendedLiveController",
    "freeze_action_manifest",
    "freeze_confirmation_manifest",
    "load_confirmation_manifest",
    "main",
    "run_extended_live",
]
