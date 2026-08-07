"""SAGE.T9.2 structural trajectory planning and frozen source-train gate."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import BindingPairRecord, load_pairs

from . import calibration_gate_v8_6 as v86
from . import calibration_gate_v8_6c as v86c
from . import calibration_gate_v8_6j_r3 as repair_r3
from . import live_shadow_pilot_v7 as live_i
from . import reachability_audit_v9 as t9_0
from . import terminal_gate_v9_1 as t9_1
from .contracts import AbstractState, ActionCandidate, ObservedTransition
from .controller import SageTConfig
from .decision import CandidateSequence, SequenceAssessment
from .posterior_v8 import T8_6G_POLICIES
from .posterior_v11 import BudgetedRepairProgramPosterior
from .replay_gate import fast_panel_from_binding_pair
from .structural_roles import (
    EASTMOST_TARGET,
    WESTMOST_TARGET,
    StructuralRoleProgramExecutor,
    augment_structural_roles,
)
from .synthesis import FragmentProposal, ProgramAssembler
from .terminal_calibration_v9 import (
    T9_1_POLICIES,
    TerminalCalibratedDecisionEngine,
    TerminalCalibratedMaterializedController,
)

FORMAT_VERSION = "sage-t9.2-trajectory-planning-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_2_trajectory_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "trajectory_v9_2"
DEFAULT_PARENT_REPORT = t9_1.DEFAULT_OUTPUT_DIR / "report.json"
SOURCE_GAMES = t9_0.SOURCE_GAMES
CHALLENGERS: Mapping[str, Mapping[str, int]] = {
    "balanced_h3": {
        "maximum_programs": 32,
        "maximum_sequences": 32,
        "maximum_particles_per_decision": 8,
        "ordinary_horizon": 3,
        "maximum_structural_macros": 8,
    },
    "wide_h3": {
        "maximum_programs": 64,
        "maximum_sequences": 64,
        "maximum_particles_per_decision": 16,
        "ordinary_horizon": 3,
        "maximum_structural_macros": 8,
    },
}


@dataclass(frozen=True)
class TrajectoryAuditRow:
    condition: str
    game: str
    root_key: str
    path: str
    remaining_path: str
    required_length: int
    structural_macros: int
    exact_sequence_generated: bool
    exact_sequence_rank: int | None
    exact_sequence_utility: float | None
    exact_sequence_terminal_risk: float | None
    exact_first_action_risk: float | None
    selected_sequence: str
    selected_first_action: bool
    selected_sequence_length: int | None
    decision_latency_ms: float
    execution_errors: int = 0


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "trajectory_planning_v9_2.py": v86c._file_sha256(
            directory / "trajectory_planning_v9_2.py"
        )
    }


def _load_parent_report(path: str | Path = DEFAULT_PARENT_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.1 report checksum mismatch")
    if report.get("status") != "T9_1_PASSED" or report.get("t9_2_authorized") is not True:
        raise ValueError("T9.1 did not authorize T9.2")
    return report


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = t9_0.DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    parent = _load_parent_report()
    selected_terminal = str(parent["selected_policy"])
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_2_SOURCE_TRAIN_GATE",
        "frozen_at": "2026-08-06",
        "parent_t9_1_report_checksum": parent["report_checksum"],
        "code_sha256": _code_hashes(),
        "shard_sha256": t9_0._shard_hashes(shard_dir),
        "source_train_games": list(SOURCE_GAMES),
        "selected_terminal_policy": selected_terminal,
        "challengers": {name: dict(caps) for name, caps in CHALLENGERS.items()},
        "structural_macros": {
            "roles": [WESTMOST_TARGET, EASTMOST_TARGET],
            "repeat_lengths": [3, 2],
            "maximum_actions": 8,
            "coordinates_are_branch_local_not_program_constants": True,
        },
        "subgoal_prior": {
            "same_target_repeat_bonus_per_extra_action": 0.35,
            "description": "persistently apply one transformation to an extremal target",
        },
        "gate": {
            "minimum_exact_sequence_generation_rate": 1.0,
            "minimum_first_action_selection_rate": 1.0,
            "minimum_exact_sequence_top8_rate": 1.0,
            "maximum_first_action_terminal_risk": 0.05,
            "maximum_decision_p95_ms": 2500.0,
            "maximum_execution_errors": 0,
        },
        "selection_order": [
            "exact_sequence_top8_rate",
            "first_action_selection_rate",
            "decision_p95_ms",
            "maximum_sequences",
        ],
        "firewall": {
            "authority": "shadow",
            "source_train_only": True,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "bounded_authority": False,
            "active_authority": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    shard_dir: str | Path = t9_0.DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.2 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.2 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_2_SOURCE_TRAIN_GATE":
        raise ValueError("T9.2 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.2 code drifted")
    if payload.get("shard_sha256") != t9_0._shard_hashes(shard_dir):
        raise ValueError("T9.2 source shards drifted")
    parent = _load_parent_report()
    if payload.get("parent_t9_1_report_checksum") != parent["report_checksum"]:
        raise ValueError("T9.2 parent report drifted")
    if payload.get("selected_terminal_policy") != parent.get("selected_policy"):
        raise ValueError("T9.2 terminal policy drifted")
    if payload.get("challengers") != {
        name: dict(caps) for name, caps in CHALLENGERS.items()
    }:
        raise ValueError("T9.2 challengers drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        bool(firewall.get(key))
        for key in (
            "source_validation_opened",
            "ar25_opened",
            "holdout_opened",
            "bounded_authority",
            "active_authority",
        )
    ):
        raise ValueError("T9.2 firewall is open")
    return payload


def structural_macros(
    state: AbstractState,
    legal_actions: Sequence[ActionCandidate],
    *,
    maximum: int = 8,
) -> tuple[tuple[ActionCandidate, ...], ...]:
    """Instantiate coordinate-free extremal-target macros in the local state."""

    enriched = augment_structural_roles(state)
    legal_names = {action.action_name for action in legal_actions}
    spatial_names = {
        action.action_name
        for action in legal_actions
        if not action.action_data or {"x", "y"}.issubset(action.action_data)
    }
    actions = sorted(legal_names & spatial_names)
    if not actions:
        return ()
    macros = []
    seen = set()
    for role in (WESTMOST_TARGET, EASTMOST_TARGET):
        entities = [
            entity
            for entity in enriched.entities_for_role(role)
            if entity.center is not None
        ]
        for entity in entities[:1]:
            row, column = entity.center or (0.0, 0.0)
            for action_name in actions:
                coordinates = {"x": round(column), "y": round(row)}
                grounded = next(
                    (
                        action
                        for action in legal_actions
                        if action.action_name == action_name
                        and all(action.action_data.get(key) == value for key, value in coordinates.items())
                    ),
                    ActionCandidate(action_name, coordinates),
                )
                for length in (3, 2):
                    macro = tuple(grounded for _ in range(length))
                    key = tuple(action.key for action in macro)
                    if key in seen:
                        continue
                    seen.add(key)
                    macros.append(macro)
                    if len(macros) >= max(0, int(maximum)):
                        return tuple(macros)
    return tuple(macros)


@dataclass
class StructuralTrajectoryDecisionEngine(TerminalCalibratedDecisionEngine):
    repeat_bonus_per_extra_action: float = 0.35

    def assess(self, *args: Any, **kwargs: Any) -> SequenceAssessment:
        assessment = super().assess(*args, **kwargs)
        actions = assessment.candidate.actions
        repeated = (
            assessment.candidate.source == "memory_macro"
            and len(actions) > 1
            and len({action.key for action in actions}) == 1
        )
        if not repeated or assessment.veto:
            return assessment
        bonus = float(self.repeat_bonus_per_extra_action) * (len(actions) - 1)
        return replace(assessment, utility=assessment.utility + bonus)


class StructuralTrajectoryController(TerminalCalibratedMaterializedController):
    """T9.2 controller adding branch-local structural macros."""

    def __init__(
        self,
        *args: Any,
        maximum_structural_macros: int = 8,
        repeat_bonus_per_extra_action: float = 0.35,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_structural_macros = int(maximum_structural_macros)
        self.repeat_bonus_per_extra_action = float(repeat_bonus_per_extra_action)
        self.decision_engine = StructuralTrajectoryDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
            calibrator=self.terminal_calibrator,
            repeat_bonus_per_extra_action=self.repeat_bonus_per_extra_action,
        )

    def _ensure_programs(self, **kwargs: Any) -> None:
        super()._ensure_programs(**kwargs)
        macros = structural_macros(
            kwargs["state"],
            kwargs["candidates"],
            maximum=self.maximum_structural_macros,
        )
        existing = tuple(self._latest_proposal.plan_sequences)
        combined = []
        seen = set()
        for macro in (*existing, *macros):
            key = tuple(action.key for action in macro)
            if key in seen:
                continue
            seen.add(key)
            combined.append(tuple(macro))
        self._latest_proposal = FragmentProposal(
            fragments=self._latest_proposal.fragments,
            plan_sequences=tuple(combined[: self.maximum_structural_macros]),
        )


def build_controller(
    caps: Mapping[str, int],
    *,
    terminal_policy_name: str,
    repeat_bonus_per_extra_action: float = 0.35,
) -> StructuralTrajectoryController:
    executor = StructuralRoleProgramExecutor()
    t7 = v86.load_t7_manifest(verify_code=True)
    posterior_policy = T8_6G_POLICIES[live_i.SELECTED_POLICY].with_repair_v2()
    config = t7["posterior"]
    posterior = BudgetedRepairProgramPosterior(
        executor=executor,
        update_policy=posterior_policy,
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(config["repair_log_likelihood_threshold"]),
        maximum_repair_contexts=repair_r3.MAXIMUM_REPAIR_CONTEXTS,
    )
    return StructuralTrajectoryController(
        executor=executor,
        posterior=posterior,
        proposer=live_i.StructuralGoalFragmentProposer(),
        assembler=ProgramAssembler(maximum_programs=int(caps["maximum_programs"])),
        config=SageTConfig(
            mode="shadow",
            maximum_programs=int(caps["maximum_programs"]),
            maximum_sequences=int(caps["maximum_sequences"]),
            maximum_particles_per_decision=int(caps["maximum_particles_per_decision"]),
            ordinary_horizon=int(caps["ordinary_horizon"]),
        ),
        terminal_policy=T9_1_POLICIES[terminal_policy_name],
        maximum_structural_macros=int(caps["maximum_structural_macros"]),
        repeat_bonus_per_extra_action=repeat_bonus_per_extra_action,
    )


def _first_action_risk(assessment: SequenceAssessment | None) -> float | None:
    if assessment is None or not assessment.predictions:
        return None
    mass = sum(item.probability for item in assessment.predictions)
    if mass <= 0.0:
        return None
    return sum(
        item.probability
        * (
            0.0
            if not item.rollout.packets
            or item.rollout.packets[0].terminal_probability is None
            else float(item.rollout.packets[0].terminal_probability)
        )
        for item in assessment.predictions
    ) / mass


def _audit_root(
    root_pairs: Sequence[BindingPairRecord],
    *,
    winning_path: str,
    condition: str,
    caps: Mapping[str, int],
    manifest: Mapping[str, Any],
) -> tuple[TrajectoryAuditRow, ...]:
    pairs_by_path = {pair.path: pair for pair in root_pairs}
    root = pairs_by_path[""]
    root_panel = fast_panel_from_binding_pair(root)
    controller = build_controller(
        caps,
        terminal_policy_name=str(manifest["selected_terminal_policy"]),
        repeat_bonus_per_extra_action=float(
            manifest["subgoal_prior"]["same_target_repeat_bonus_per_extra_action"]
        ),
    )
    history: list[ObservedTransition] = []
    names = tuple(sorted({arm.action.action_name for arm in root_panel.arms}))

    def assemble(*, seed: bool, state: AbstractState) -> None:
        proposal = controller.proposer.propose(
            available_actions=names,
            transitions=tuple(history),
        )
        programs = controller.assembler.assemble(
            proposal.fragments,
            available_actions=names,
        )
        if seed:
            controller.posterior.seed(programs, initial_state=state)
        else:
            controller.posterior.add_programs(programs, initial_state=state)

    assemble(seed=True, state=root_panel.state)
    for event in root.context:
        evidence = t9_0._context_transition(event, root_panel.state)
        controller.posterior.observe(evidence)
        controller.terminal_calibrator.observe(evidence)
        history.append(evidence)
        assemble(seed=False, state=root_panel.state)

    rows = []
    prefix = ""
    for _ in winning_path:
        pair = pairs_by_path[prefix]
        panel = fast_panel_from_binding_pair(pair)
        legal = tuple(arm.action for arm in panel.arms)
        remaining = t9_0._remaining_actions(
            pairs_by_path,
            current_path=prefix,
            winning_path=winning_path,
        )
        expected = CandidateSequence(remaining, source="oracle")
        macros = structural_macros(
            panel.state,
            legal,
            maximum=int(caps["maximum_structural_macros"]),
        )
        started = time.perf_counter()
        try:
            decision = controller.decision_engine.decide(
                controller.posterior,
                panel.state,
                legal,
                memory_macros=macros,
            )
            errors = 0
        except (ArithmeticError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            decision = None
            errors = 1
        latency = (time.perf_counter() - started) * 1000.0
        assessments = () if decision is None else decision.assessments
        match = next(
            (item for item in assessments if item.candidate.key == expected.key),
            None,
        )
        rank = next(
            (
                index
                for index, item in enumerate(assessments, start=1)
                if item.candidate.key == expected.key
            ),
            None,
        )
        chosen = None if decision is None else decision.chosen
        first_risk = _first_action_risk(match)
        if first_risk is not None:
            first_risk = controller.terminal_calibrator.calibrate(
                remaining[0],
                first_risk,
                regime_index=panel.state.regime_index,
            )
        rows.append(
            TrajectoryAuditRow(
                condition=condition,
                game=pair.game_id,
                root_key=pair.root_key,
                path=prefix,
                remaining_path=winning_path[len(prefix) :],
                required_length=len(remaining),
                structural_macros=len(macros),
                exact_sequence_generated=match is not None,
                exact_sequence_rank=rank,
                exact_sequence_utility=None if match is None else match.utility,
                exact_sequence_terminal_risk=(
                    None if match is None else match.terminal_risk
                ),
                exact_first_action_risk=first_risk,
                selected_sequence="" if chosen is None else chosen.candidate.key,
                selected_first_action=bool(
                    chosen is not None and chosen.first_action.key == remaining[0].key
                ),
                selected_sequence_length=(
                    None if chosen is None else len(chosen.candidate.actions)
                ),
                decision_latency_ms=latency,
                execution_errors=errors,
            )
        )
        symbol = winning_path[len(prefix)]
        observed = panel.arms[0 if symbol == "L" else 1]
        controller.posterior.observe(observed)
        controller.terminal_calibrator.observe(observed)
        history.append(observed)
        assemble(seed=False, state=observed.state_after)
        prefix += symbol
    return tuple(rows)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("inf")
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


def build_report(
    rows: Sequence[TrajectoryAuditRow],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    conditions = {}
    survivors = []
    gate = manifest["gate"]
    for name, caps in CHALLENGERS.items():
        selected = [row for row in rows if row.condition == name]
        count = len(selected)
        generation = mean(row.exact_sequence_generated for row in selected)
        first = mean(row.selected_first_action for row in selected)
        top8 = mean(
            row.exact_sequence_rank is not None and row.exact_sequence_rank <= 8
            for row in selected
        )
        risks = [
            float(row.exact_first_action_risk)
            for row in selected
            if row.exact_first_action_risk is not None
        ]
        p95 = _quantile([row.decision_latency_ms for row in selected], 0.95)
        checks = {
            "nine_prefixes": count == 9,
            "exact_sequence_generation": generation
            >= float(gate["minimum_exact_sequence_generation_rate"]),
            "first_action_selection": first
            >= float(gate["minimum_first_action_selection_rate"]),
            "exact_sequence_top8": top8
            >= float(gate["minimum_exact_sequence_top8_rate"]),
            "first_action_terminal_risk": bool(risks)
            and max(risks) <= float(gate["maximum_first_action_terminal_risk"]) + 1e-12,
            "decision_p95": p95 <= float(gate["maximum_decision_p95_ms"]),
            "zero_execution_errors": sum(row.execution_errors for row in selected)
            <= int(gate["maximum_execution_errors"]),
            "macro_limit": all(row.structural_macros <= 8 for row in selected),
        }
        conditions[name] = {
            "caps": dict(caps),
            "rows": count,
            "exact_sequence_generation_rate": generation,
            "first_action_selection_rate": first,
            "exact_sequence_top8_rate": top8,
            "maximum_first_action_terminal_risk": max(risks, default=None),
            "decision_p95_ms": p95,
            "mean_selected_sequence_length": mean(
                row.selected_sequence_length or 0 for row in selected
            ),
            "checks": checks,
            "passed": all(checks.values()),
        }
        if all(checks.values()):
            survivors.append(name)
    selected_name = (
        min(
            survivors,
            key=lambda name: (
                -conditions[name]["exact_sequence_top8_rate"],
                -conditions[name]["first_action_selection_rate"],
                conditions[name]["decision_p95_ms"],
                CHALLENGERS[name]["maximum_sequences"],
            ),
        )
        if survivors
        else ""
    )
    passed = bool(selected_name)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T9_2_PASSED" if passed else "T9_2_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "selected_challenger": selected_name or None,
        "conditions": conditions,
        "t9_3_authorized": passed,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    payload["report_checksum"] = v86c._checksum(payload)
    return payload


def run_gate(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = t9_0.DEFAULT_SHARD_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, shard_dir=shard_dir)
    pairs = load_pairs(shard_dir, SOURCE_GAMES)
    paths = t9_0.winner_paths(pairs)
    grouped: dict[str, list[BindingPairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    rows = tuple(
        row
        for condition, caps in CHALLENGERS.items()
        for root_key, winning_paths in sorted(paths.items())
        for row in _audit_root(
            grouped[root_key],
            winning_path=winning_paths[0],
            condition=condition,
            caps=caps,
            manifest=manifest,
        )
    )
    report = build_report(rows, manifest=manifest)
    destination = Path(output_dir)
    v86c._write_jsonl(destination / "rows.jsonl", (asdict(row) for row in rows))
    v86c._write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--shard-dir", default=str(t9_0.DEFAULT_SHARD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(output_path=args.manifest, shard_dir=args.shard_dir)
    else:
        result = run_gate(
            manifest_path=args.manifest,
            shard_dir=args.shard_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_2_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHALLENGERS",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "StructuralTrajectoryController",
    "StructuralTrajectoryDecisionEngine",
    "TrajectoryAuditRow",
    "build_controller",
    "build_report",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
    "structural_macros",
]
