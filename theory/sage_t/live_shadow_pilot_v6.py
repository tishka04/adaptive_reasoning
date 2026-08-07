"""T8.6 shared-action live confirmation for calibrated posterior policies."""

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

from . import live_shadow_pilot_v5 as t8_5
from .calibration_gate_v8_6 import (
    CONFIRMATION_FORMAT_VERSION,
    DEFAULT_CONFIRMATION_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTION_MANIFEST,
    _checksum,
    _code_hashes,
    _write_json,
    _write_jsonl,
    load_selection_manifest,
)
from .controller import SageTConfig
from .executor import ProgramExecutor
from .live_shadow_pilot_v3 import assessment_for_live_action
from .posterior_v2 import (
    T8_6_POLICIES,
    CalibratedProgramPosterior,
    PosteriorUpdatePolicy,
)

DEFAULT_SELECTION_REPORT = DEFAULT_OUTPUT_DIR / "selection_report.json"
DEFAULT_LIVE_OUTPUT_DIR = Path("training") / "sage_t" / "calibration_v8_6_live"


def load_confirmation_manifest(
    path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    *,
    selection_report_path: str | Path = DEFAULT_SELECTION_REPORT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T8.6 confirmation manifest checksum mismatch")
    if payload.get("format_version") != CONFIRMATION_FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.6 confirmation manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_6_LIVE_CONFIRMATION":
        raise ValueError("SAGE.T8.6 confirmation manifest is not frozen")
    if payload.get("authority") != "shadow":
        raise ValueError("T8.6 live confirmation must remain shadow")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("SAGE.T8.6 code drifted after confirmation freeze")
    report = json.loads(Path(selection_report_path).read_text(encoding="utf-8"))
    report_checksum = str(report.get("report_checksum", ""))
    unsigned_report = dict(report)
    unsigned_report.pop("report_checksum", None)
    if report_checksum != _checksum(unsigned_report):
        raise ValueError("SAGE.T8.6 selection report checksum mismatch")
    if payload.get("selection_report_checksum") != report_checksum:
        raise ValueError("confirmation does not bind the selection report")
    if payload.get("selected_challenger") != report.get("selected_challenger"):
        raise ValueError("confirmation challenger drifted")
    if report.get("source_validation_authorized") is not False:
        raise ValueError("source-validation firewall opened before confirmation")
    return payload, report


def _controller(
    policy: PosteriorUpdatePolicy,
    *,
    caps: Mapping[str, Any],
) -> t8_5.MaterializedActionController:
    executor = ProgramExecutor()
    posterior = CalibratedProgramPosterior(
        executor=executor,
        update_policy=policy,
        maximum_particles=int(caps["maximum_programs"]),
    )
    return t8_5.MaterializedActionController(
        executor=executor,
        posterior=posterior,
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


class MultiPolicyMaterializedController:
    """Broadcast one baseline action context and transition to every policy."""

    def __init__(
        self,
        *,
        selected: str,
        caps: Mapping[str, Any],
        include_repair_v2: bool = True,
    ) -> None:
        policies = dict(T8_6_POLICIES)
        selected_v2 = T8_6_POLICIES[selected].with_repair_v2()
        if include_repair_v2:
            policies[selected_v2.name] = selected_v2
        self.controllers = {
            name: _controller(policy, caps=caps)
            for name, policy in policies.items()
        }
        self.selected_name = selected_v2.name if include_repair_v2 else selected
        self.selected = self.controllers[self.selected_name]
        self.posterior = self.selected.posterior

    @property
    def effective_mode(self):  # type: ignore[no-untyped-def]
        return self.selected.effective_mode

    def decide(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        arbitrations = {
            name: controller.decide(**kwargs)
            for name, controller in self.controllers.items()
        }
        return arbitrations[self.selected_name]

    def observe_transition(self, record: Any) -> None:
        for controller in self.controllers.values():
            controller.observe_transition(record)

    def start_branch(self, *, regime_index: int | None = None) -> None:
        for controller in self.controllers.values():
            controller.start_branch(regime_index=regime_index)

    def note_level_change(self) -> None:
        for controller in self.controllers.values():
            controller.note_level_change()

    def summary(self) -> Mapping[str, Any]:
        payload = dict(self.selected.summary())
        payload["t8_6_selected_policy"] = self.selected_name
        payload["t8_6_policy_summaries"] = {
            name: controller.summary()
            for name, controller in self.controllers.items()
        }
        return payload


def _factory_builder(
    *,
    selected: str,
    registry: dict[str, MultiPolicyMaterializedController],
    include_repair_v2: bool = True,
) -> Any:
    def builder(*, mode: str, manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
        caps = manifest["controller"]

        def factory(game_id: str) -> UnifiedCognitiveController:
            if mode == "off":
                return UnifiedCognitiveController(
                    game_id,
                    config=UnifiedCognitiveConfig(sage_t_authority_mode="off"),
                )
            sage_t = MultiPolicyMaterializedController(
                selected=selected,
                caps=caps,
                include_repair_v2=include_repair_v2,
            )
            registry[str(game_id)] = sage_t
            return UnifiedCognitiveController(
                game_id,
                config=UnifiedCognitiveConfig(sage_t_authority_mode="shadow"),
                sage_t_controller=sage_t,  # type: ignore[arg-type]
            )

        return factory

    return builder


def run_live_negative_replay(
    *,
    selection_manifest_path: str | Path = DEFAULT_SELECTION_MANIFEST,
    t8_5_manifest_path: str | Path = t8_5.DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Replay the 50 T8.5 negative actions through all four policies."""

    manifest = load_selection_manifest(selection_manifest_path)
    registry: dict[str, MultiPolicyMaterializedController] = {}
    previous_factory = t8_5._controller_factory
    t8_5._controller_factory = _factory_builder(
        selected="legacy",
        registry=registry,
        include_repair_v2=False,
    )
    try:
        base_report = t8_5.run_live_shadow_pilot(
            manifest_path=t8_5_manifest_path,
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
    policy_rows = _policy_live_rows(registry, base_rows)
    by_policy = {}
    for name in T8_6_POLICIES:
        selected = [row for row in policy_rows if row["condition"] == name]
        by_policy[name] = {
            "actions": len(selected),
            "prediction_coverage": mean(
                float(bool(row["prediction_available"])) for row in selected
            ) if selected else 0.0,
            "semantic_collapse_rate": mean(
                float(bool(row["semantic_collapse"])) for row in selected
            ) if selected else 0.0,
            "raw_mixture_surprise": mean(
                float(row["raw_mixture_surprise"])
                for row in selected
                if row.get("raw_mixture_surprise") is not None
            ) if any(
                row.get("raw_mixture_surprise") is not None for row in selected
            ) else None,
        }
    report: dict[str, Any] = {
        "format_version": CONFIRMATION_FORMAT_VERSION,
        "status": "LIVE_NEGATIVE_REPLAY_COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "actions": len(base_rows),
        "policies": by_policy,
        "same_actions": bool(
            base_report.get("safety", {}).get("actions_match")
        ),
        "zero_interventions": int(
            base_report.get("safety", {}).get("interventions", 0)
        ) == 0,
        "source_validation_authorized": False,
        "authority_authorized": False,
    }
    report["report_checksum"] = _checksum(report)
    _write_jsonl(destination / "live_negative_policy_rows.jsonl", policy_rows)
    _write_json(destination / "live_negative_report.json", report)
    return report


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _policy_live_rows(
    registry: Mapping[str, MultiPolicyMaterializedController],
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for item in base_rows:
        by_game.setdefault(str(item["game_id"]), []).append(item)
    for game, multi in registry.items():
        actual = by_game.get(game, ())
        for name, controller in multi.controllers.items():
            decisions = [
                item
                for item in controller.compact_records
                if item.get("kind") == "decision"
            ]
            observations = [
                item
                for item in controller.compact_records
                if item.get("kind") == "observation"
            ]
            for index, truth in enumerate(actual):
                decision = decisions[index] if index < len(decisions) else {}
                observation = (
                    observations[index] if index < len(observations) else {}
                )
                assessment = assessment_for_live_action(
                    decision,
                    str(truth["action_key"]),
                )
                after = dict(observation.get("posterior_after", {}) or {})
                diagnostics = dict(after.get("last_update", {}) or {})
                rows.append(
                    {
                        "game": game,
                        "condition": name,
                        "step": index,
                        "action_key": truth["action_key"],
                        "actual_terminal": int(truth["actual_terminal"]),
                        "actual_goal": int(truth["actual_goal"]),
                        "actual_progress": float(truth["actual_progress"]),
                        "prediction_available": assessment is not None,
                        "terminal_probability": (
                            None
                            if assessment is None
                            else assessment.get("terminal_risk")
                        ),
                        "goal_probability": (
                            None
                            if assessment is None
                            else assessment.get("expected_goal")
                        ),
                        "progress_prediction": (
                            None
                            if assessment is None
                            else assessment.get("expected_progress")
                        ),
                        "decision_latency_ms": (
                            None
                            if index >= len(controller.decision_latencies_ms)
                            else controller.decision_latencies_ms[index]
                        ),
                        "observation_latency_ms": (
                            None
                            if index >= len(controller.observation_latencies_ms)
                            else controller.observation_latencies_ms[index]
                        ),
                        "semantic_collapse": bool(
                            diagnostics.get("semantic_collapse", False)
                        ),
                        "raw_mixture_surprise": diagnostics.get(
                            "raw_mixture_surprise"
                        ),
                        "repair_cycles_delta": diagnostics.get(
                            "repair_cycle_delta",
                            0,
                        ),
                        "repair_survived_delta": diagnostics.get(
                            "repair_survived_delta",
                            0,
                        ),
                    }
                )
    return rows


def _live_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_condition: str,
    base_report: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == selected_condition]
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
    coverage = mean(
        float(bool(row["prediction_available"])) for row in selected
    ) if selected else 0.0
    config = confirmation.get("gate", {})
    decision_limit = float(config.get("maximum_decision_p95_ms", 2500.0))
    observation_limit = float(config.get("maximum_observation_p95_ms", 3000.0))
    first = decisions[:5]
    last = decisions[-5:]
    tail_ratio = mean(last) / max(1e-12, mean(first)) if first and last else math.inf
    repair_budget = all(
        int(row.get("repair_cycles_delta", 0)) <= 1
        and int(row.get("repair_survived_delta", 0)) <= 4
        for row in selected
    )
    safety = dict(base_report.get("safety", {}) or {})
    checks = {
        "fifty_actions": len(selected) == int(confirmation["actions"]),
        "prediction_coverage": coverage == 1.0,
        "same_actions": bool(safety.get("actions_match")),
        "zero_interventions": int(safety.get("interventions", 0)) == 0,
        "zero_illegal_actions": int(safety.get("illegal_actions", 0)) == 0,
        "zero_controller_errors": int(safety.get("controller_errors", 0)) == 0,
        "decision_p95": bool(decisions)
        and _quantile(decisions, 0.95) <= decision_limit,
        "observation_p95": bool(observations)
        and _quantile(observations, 0.95) <= observation_limit,
        "live_wall_time": float(base_report.get("wall_clock_seconds", math.inf))
        <= 100.0,
        "latency_tail_ratio": tail_ratio <= 4.0,
        "repair_budgets": repair_budget,
        "no_semantic_collapse": not any(
            bool(row["semantic_collapse"]) for row in selected
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "actions": len(selected),
            "prediction_coverage": coverage,
            "decision_p95_ms": _quantile(decisions, 0.95),
            "observation_p95_ms": _quantile(observations, 0.95),
            "latency_tail_ratio": tail_ratio,
            "wall_clock_seconds": base_report.get("wall_clock_seconds"),
            "semantic_collapses": sum(
                bool(row["semantic_collapse"]) for row in selected
            ),
        },
    }


def run_live_confirmation(
    *,
    confirmation_manifest_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    selection_report_path: str | Path = DEFAULT_SELECTION_REPORT,
    t8_5_manifest_path: str | Path = t8_5.DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_LIVE_OUTPUT_DIR,
) -> dict[str, Any]:
    confirmation, selection = load_confirmation_manifest(
        confirmation_manifest_path,
        selection_report_path=selection_report_path,
    )
    registry: dict[str, MultiPolicyMaterializedController] = {}
    previous_factory = t8_5._controller_factory
    t8_5._controller_factory = _factory_builder(
        selected=str(confirmation["selected_challenger"]),
        registry=registry,
    )
    started = time.perf_counter()
    try:
        base_report = t8_5.run_live_shadow_pilot(
            manifest_path=t8_5_manifest_path,
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
    policy_rows = _policy_live_rows(registry, base_rows)
    selected_condition = str(
        confirmation["repair_policy"]["name"]
    )
    live = _live_checks(
        policy_rows,
        selected_condition=selected_condition,
        base_report=base_report,
        confirmation={
            **confirmation,
            "gate": selection.get("confirmation_gate", {}),
        },
    )
    offline = dict(selection.get("offline_repair_v2_confirmation") or {})
    passed = bool(offline.get("passed")) and bool(live.get("passed"))
    conclusion = (
        "CALIBRATION_RECOVERED"
        if passed
        else str(selection.get("conclusion", "INCONCLUSIVE_FAIL_CLOSED"))
    )
    if conclusion == "CALIBRATION_RECOVERED" and not passed:
        conclusion = "INCONCLUSIVE_FAIL_CLOSED"
    report: dict[str, Any] = {
        "format_version": CONFIRMATION_FORMAT_VERSION,
        "status": (
            "READY_TO_PREPARE_T8_7_SOURCE_VALIDATION"
            if passed
            else "T8_6_CONFIRMATION_FAILED_CLOSED"
        ),
        "manifest_checksum": confirmation["manifest_checksum"],
        "selection_report_checksum": selection["report_checksum"],
        "selected_challenger": confirmation["selected_challenger"],
        "repair_policy": confirmation["repair_policy"],
        "offline_confirmation": offline,
        "live_confirmation": live,
        "base_live_report_checksum": base_report.get("report_checksum"),
        "elapsed_seconds": time.perf_counter() - started,
        "conclusion": conclusion,
        "source_validation_authorized": passed,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
    }
    report["report_checksum"] = _checksum(report)
    _write_jsonl(destination / "policy_rows.jsonl", policy_rows)
    _write_json(destination / "t8_6_confirmation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-manifest", default=str(DEFAULT_CONFIRMATION_MANIFEST))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION_MANIFEST))
    parser.add_argument("--selection-report", default=str(DEFAULT_SELECTION_REPORT))
    parser.add_argument("--t8-5-manifest", default=str(t8_5.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_LIVE_OUTPUT_DIR))
    parser.add_argument("--live-negative", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.live_negative:
        report = run_live_negative_replay(
            selection_manifest_path=args.selection_manifest,
            t8_5_manifest_path=args.t8_5_manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("same_actions") else 2
    report = run_live_confirmation(
        confirmation_manifest_path=args.confirmation_manifest,
        selection_report_path=args.selection_report,
        t8_5_manifest_path=args.t8_5_manifest,
        environments_dir=args.environments_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("source_validation_authorized") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LIVE_OUTPUT_DIR",
    "DEFAULT_SELECTION_REPORT",
    "MultiPolicyMaterializedController",
    "load_confirmation_manifest",
    "main",
    "run_live_confirmation",
    "run_live_negative_replay",
]
