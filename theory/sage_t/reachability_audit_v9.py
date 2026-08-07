"""SAGE.T9.0 source-train reachability audit on frozen V4.3 trees.

The audit follows every observed level-winning path and assigns one exclusive
failure diagnosis to each of its prefixes.  Contexts for which the depth-three
tree contains no level signal remain accounted for, but are explicitly marked
as ground-truth-uncovered rather than being used to judge the model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from theory.sage12.bound_mechanic_pilot import BindingPairRecord, load_pairs

from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot_v10 as live_r3
from . import live_shadow_pilot_v11 as t8_7
from .contracts import (
    AbstractState,
    ActionCandidate,
    ObservedTransition,
    PredictionPacket,
    RolloutPrediction,
)
from .decision import CandidateSequence, SequenceAssessment
from .replay_gate import fast_panel_from_binding_pair
from .synthesis import AssembledProgram, ProgramAssembler

FORMAT_VERSION = "sage-t9.0-reachability-audit-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_0_reachability_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "reachability_v9_0"
DEFAULT_SHARD_DIR = (
    Path("training")
    / "sage12"
    / "bound_mechanic_pilot_v4_3"
    / "source_train_shards"
)
SOURCE_GAMES = ("lp85", "su15")
CURRENT_CAPS = {
    "maximum_programs": 8,
    "maximum_sequences": 8,
    "maximum_particles_per_decision": 4,
    "ordinary_horizon": 1,
}
DIAGNOSES = (
    "NO_GOAL_CANDIDATE",
    "GOAL_CANDIDATE_PRUNED",
    "GOAL_CANDIDATE_UNDERVALUED",
    "GOAL_CANDIDATE_VETOED",
    "EXECUTION_MODEL_MISS",
    "REACHABLE",
)


@dataclass(frozen=True)
class ReachabilityRow:
    game: str
    root_key: str
    pair_digest: str
    path: str
    depth: int
    ground_truth_covered: bool
    winning_path: str
    remaining_path: str
    required_sequence_length: int | None
    requires_multi_action: bool
    goal_action_generated: bool
    goal_sequence_generated: bool
    unpruned_static_goal_programs: int
    unpruned_executable_goal_programs: int
    production_executable_goal_programs: int
    goal_program_rank: int | None
    goal_program_mass: float
    goal_action_rank: int | None
    goal_action_in_top8: bool
    selected_goal_action: bool
    goal_action_terminal_risk: float | None
    goal_sequence_terminal_risk: float | None
    diagnosis: str
    reason: str
    execution_errors: int = 0


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "reachability_audit_v9.py": v86c._file_sha256(
            directory / "reachability_audit_v9.py"
        )
    }


def _passed_t8_7_report(
    path: str | Path = t8_7.DEFAULT_OUTPUT_DIR / "t8_7_source_validation_report.json",
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.7 report checksum mismatch")
    if report.get("status") != "T8_7_PASSED":
        raise ValueError("T8.7 did not pass")
    if report.get("t9_audit_authorized") is not True:
        raise ValueError("T8.7 did not authorize T9.0")
    if report.get("bounded_authority_authorized") is not False:
        raise ValueError("authority opened before T9.0")
    return report


def _shard_hashes(shard_dir: str | Path = DEFAULT_SHARD_DIR) -> dict[str, str]:
    root = Path(shard_dir)
    return {
        f"{game}.jsonl": v86c._file_sha256(root / f"{game}.jsonl")
        for game in SOURCE_GAMES
    }


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    t8_7_report = _passed_t8_7_report()
    action_manifest = json.loads(
        live_r3.DEFAULT_ACTION_MANIFEST.read_text(encoding="utf-8")
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_0_SOURCE_TRAIN_AUDIT",
        "frozen_at": "2026-08-06",
        "parent_t8_7_report_checksum": t8_7_report["report_checksum"],
        "t8_6j_action_manifest_checksum": action_manifest["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "shard_sha256": _shard_hashes(shard_dir),
        "source_train_games": list(SOURCE_GAMES),
        "controller_caps": dict(CURRENT_CAPS),
        "unpruned_caps": {
            "maximum_programs": 512,
            "maximum_dynamics_beam": 64,
        },
        "terminal_veto_threshold": 0.8,
        "diagnoses": list(DIAGNOSES),
        "firewall": {
            "authority": "shadow",
            "source_train_only": True,
            "source_validation_read_for_parent_gate_only": True,
            "source_validation_games_executed": False,
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
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.0 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.0 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_0_SOURCE_TRAIN_AUDIT":
        raise ValueError("T9.0 manifest is not frozen")
    if tuple(payload.get("source_train_games", ())) != SOURCE_GAMES:
        raise ValueError("T9.0 source games drifted")
    if payload.get("controller_caps") != CURRENT_CAPS:
        raise ValueError("T9.0 controller caps drifted")
    if payload.get("shard_sha256") != _shard_hashes(shard_dir):
        raise ValueError("T9.0 source-train shards drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.0 code drifted")
    parent = _passed_t8_7_report()
    if payload.get("parent_t8_7_report_checksum") != parent["report_checksum"]:
        raise ValueError("T9.0 parent report drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        bool(firewall.get(key))
        for key in (
            "source_validation_games_executed",
            "ar25_opened",
            "holdout_opened",
            "bounded_authority",
            "active_authority",
        )
    ):
        raise ValueError("T9.0 firewall is open")
    return payload


def _level_positive(pair: BindingPairRecord, side: str) -> bool:
    trace = getattr(pair, side).trace
    return bool(trace.effects.level_complete) or (
        int(trace.levels_completed_after) > int(trace.levels_completed_before)
    )


def winner_paths(
    pairs: Sequence[BindingPairRecord],
) -> dict[str, tuple[str, ...]]:
    paths: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        for side, symbol in (("left", "L"), ("right", "R")):
            if _level_positive(pair, side):
                paths[pair.root_key].add(pair.path + symbol)
    return {
        root: tuple(sorted(values, key=lambda value: (len(value), value)))
        for root, values in paths.items()
    }


def _context_transition(event: Any, state: AbstractState) -> ObservedTransition:
    events = tuple(
        sorted(
            name.removeprefix("target_")
            for name, value in dict(event.effects).items()
            if bool(value)
        )
    )
    if not events:
        events = ("no_effect",)
    packet = PredictionPacket(
        object_deltas={name: 1.0 for name in events},
        terminal_probability=0.0,
        known_channels=frozenset({"objects", "terminal"}),
        state_after=state,
    )
    return ObservedTransition(
        state_before=state,
        action=ActionCandidate(str(event.action_name)),
        state_after=state,
        observation=packet,
        events=events,
    )


def _goal_signal(rollout: RolloutPrediction) -> bool:
    return any(
        (packet.goal_probability is not None and packet.goal_probability >= 0.5)
        or (packet.progress_mean is not None and packet.progress_mean > 0.0)
        for packet in rollout.packets
    )


def _declares_goal_effect(assembled: AssembledProgram) -> bool:
    family = assembled.program.goal_rule.family
    expected = {
        "level_completion": {("assert", "level_complete"), ("progress", "")},
        "solve_all_targets": {("assert", "solved")},
        "reach_target": {("assert", "contact"), ("move_relative", "")},
        "exhaust_targets": {("remove", ""), ("assert", "removed")},
    }.get(family, set())
    return any(
        (effect.operation, effect.predicate) in expected
        or (effect.operation, "") in expected
        for rule in assembled.program.transition_rules
        for effect in rule.effects
    )


def _rollout_goal_programs(
    programs: Sequence[Any],
    *,
    executor: Any,
    state: AbstractState,
    actions: Sequence[ActionCandidate],
) -> tuple[list[Any], int]:
    matches = []
    errors = 0
    for item in programs:
        program = getattr(item, "program", item)
        try:
            rollout = executor.rollout(program, state, actions, maximum_actions=8)
        except (ArithmeticError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            errors += 1
            continue
        if _goal_signal(rollout):
            matches.append(item)
    return matches, errors


def _remaining_actions(
    pairs_by_path: Mapping[str, BindingPairRecord],
    *,
    current_path: str,
    winning_path: str,
) -> tuple[ActionCandidate, ...]:
    actions = []
    prefix = current_path
    for symbol in winning_path[len(current_path) :]:
        pair = pairs_by_path[prefix]
        panel = fast_panel_from_binding_pair(pair)
        actions.append(panel.arms[0 if symbol == "L" else 1].action)
        prefix += symbol
    return tuple(actions)


def classify_failure(
    *,
    ground_truth_covered: bool,
    goal_sequence_generated: bool,
    unpruned_static: int,
    unpruned_executable: int,
    production_executable: int,
    goal_program_rank: int | None,
    selected_goal_action: bool,
    terminal_blocked: bool,
    execution_errors: int,
) -> tuple[str, str]:
    if not ground_truth_covered:
        return "NO_GOAL_CANDIDATE", "no_observed_goal_within_depth_three"
    if not goal_sequence_generated:
        return "NO_GOAL_CANDIDATE", "required_sequence_not_generated"
    if unpruned_static <= 0:
        return "NO_GOAL_CANDIDATE", "grammar_has_no_compatible_goal_effect"
    if unpruned_executable <= 0:
        return "EXECUTION_MODEL_MISS", (
            "static_goal_program_does_not_execute"
            if execution_errors <= 0
            else "goal_program_execution_error"
        )
    if production_executable <= 0:
        return "GOAL_CANDIDATE_PRUNED", "candidate_lost_in_eight_program_beam"
    if goal_program_rank is None or goal_program_rank > 8:
        return "GOAL_CANDIDATE_UNDERVALUED", "candidate_absent_from_posterior_top8"
    if terminal_blocked:
        return "GOAL_CANDIDATE_VETOED", "terminal_penalty_blocks_goal_action"
    if not selected_goal_action:
        return "GOAL_CANDIDATE_UNDERVALUED", "goal_action_not_selected"
    return "REACHABLE", "goal_action_selected"


def _assessment_for_action(
    assessments: Sequence[SequenceAssessment],
    action: ActionCandidate,
) -> tuple[int | None, SequenceAssessment | None]:
    for index, assessment in enumerate(assessments, start=1):
        if assessment.first_action.key == action.key:
            return index, assessment
    return None, None


def _audit_winning_root(
    root_pairs: Sequence[BindingPairRecord],
    *,
    winning_path: str,
    manifest: Mapping[str, Any],
) -> tuple[ReachabilityRow, ...]:
    pairs_by_path = {pair.path: pair for pair in root_pairs}
    root = pairs_by_path[""]
    root_panel = fast_panel_from_binding_pair(root)
    selected = live_r3._controller(caps=manifest["controller_caps"])
    posterior = selected.posterior
    executor = selected.executor
    proposer = selected.proposer
    assembler = selected.assembler
    engine = selected.decision_engine
    available_names = tuple(
        sorted({arm.action.action_name for arm in root_panel.arms})
    )
    history: list[ObservedTransition] = []

    def assemble(*, seed: bool, state: AbstractState) -> tuple[AssembledProgram, ...]:
        proposal = proposer.propose(
            available_actions=available_names,
            transitions=tuple(history),
        )
        programs = assembler.assemble(
            proposal.fragments,
            available_actions=available_names,
        )
        if seed:
            posterior.seed(programs, initial_state=state)
        else:
            posterior.add_programs(programs, initial_state=state)
        return programs

    assemble(seed=True, state=root_panel.state)
    for event in root.context:
        evidence = _context_transition(event, root_panel.state)
        posterior.observe(evidence)
        history.append(evidence)
        assemble(seed=False, state=root_panel.state)

    rows = []
    prefix = ""
    for _ in winning_path:
        pair = pairs_by_path[prefix]
        panel = fast_panel_from_binding_pair(pair)
        remaining = _remaining_actions(
            pairs_by_path,
            current_path=prefix,
            winning_path=winning_path,
        )
        first = remaining[0]
        proposal = proposer.propose(
            available_actions=available_names,
            transitions=tuple(history),
        )
        production = assembler.assemble(
            proposal.fragments,
            available_actions=available_names,
        )
        unpruned = ProgramAssembler(
            maximum_programs=int(manifest["unpruned_caps"]["maximum_programs"]),
            maximum_dynamics_beam=int(
                manifest["unpruned_caps"]["maximum_dynamics_beam"]
            ),
        ).assemble(proposal.fragments, available_actions=available_names)
        static = sum(_declares_goal_effect(item) for item in unpruned)
        unpruned_goal, unpruned_errors = _rollout_goal_programs(
            unpruned,
            executor=executor,
            state=panel.state,
            actions=remaining,
        )
        production_goal, production_errors = _rollout_goal_programs(
            production,
            executor=executor,
            state=panel.state,
            actions=remaining,
        )
        ranked = posterior.top(len(posterior.particles))
        posterior_goal, posterior_errors = _rollout_goal_programs(
            ranked,
            executor=executor,
            state=panel.state,
            actions=remaining,
        )
        goal_hashes = {
            item.program.canonical_hash for item in posterior_goal
        }
        goal_rank = next(
            (
                index
                for index, particle in enumerate(ranked, start=1)
                if particle.program.canonical_hash in goal_hashes
            ),
            None,
        )
        goal_mass = sum(
            particle.probability
            for particle in ranked
            if particle.program.canonical_hash in goal_hashes
        )
        decision = engine.decide(
            posterior,
            panel.state,
            tuple(arm.action for arm in panel.arms),
        )
        action_rank, action_assessment = _assessment_for_action(
            decision.assessments,
            first,
        )
        selected_goal = bool(
            decision.action is not None and decision.action.key == first.key
        )
        planner_sequences = engine.generate_sequences(
            tuple(arm.action for arm in panel.arms)
        )
        sequence = CandidateSequence(remaining, source="oracle_audit")
        sequence_generated = any(
            candidate.key == sequence.key for candidate in planner_sequences
        )
        sequence_assessment = engine.assess(
            sequence,
            particles=posterior.top(engine.maximum_particles),
            state=panel.state,
            beta=1.0 if posterior.normalized_entropy > 0.5 else 0.25,
        )
        action_risk = (
            None if action_assessment is None else action_assessment.terminal_risk
        )
        threshold = float(manifest["terminal_veto_threshold"])
        terminal_blocked = bool(
            not selected_goal
            and action_risk is not None
            and action_risk >= threshold
        )
        errors = unpruned_errors + production_errors + posterior_errors
        diagnosis, reason = classify_failure(
            ground_truth_covered=True,
            goal_sequence_generated=sequence_generated,
            unpruned_static=static,
            unpruned_executable=len(unpruned_goal),
            production_executable=len(production_goal),
            goal_program_rank=goal_rank,
            selected_goal_action=selected_goal,
            terminal_blocked=terminal_blocked,
            execution_errors=errors,
        )
        rows.append(
            ReachabilityRow(
                game=pair.game_id,
                root_key=pair.root_key,
                pair_digest=pair.pair_digest,
                path=prefix,
                depth=pair.depth,
                ground_truth_covered=True,
                winning_path=winning_path,
                remaining_path=winning_path[len(prefix) :],
                required_sequence_length=len(remaining),
                requires_multi_action=len(remaining) > 1,
                goal_action_generated=any(
                    arm.action.key == first.key for arm in panel.arms
                ),
                goal_sequence_generated=sequence_generated,
                unpruned_static_goal_programs=static,
                unpruned_executable_goal_programs=len(unpruned_goal),
                production_executable_goal_programs=len(production_goal),
                goal_program_rank=goal_rank,
                goal_program_mass=goal_mass,
                goal_action_rank=action_rank,
                goal_action_in_top8=action_rank is not None and action_rank <= 8,
                selected_goal_action=selected_goal,
                goal_action_terminal_risk=action_risk,
                goal_sequence_terminal_risk=sequence_assessment.terminal_risk,
                diagnosis=diagnosis,
                reason=reason,
                execution_errors=errors,
            )
        )
        symbol = winning_path[len(prefix)]
        chosen = panel.arms[0 if symbol == "L" else 1]
        posterior.observe(chosen)
        history.append(chosen)
        assemble(seed=False, state=chosen.state_after)
        prefix += symbol
    return tuple(rows)


def _uncovered_row(pair: BindingPairRecord) -> ReachabilityRow:
    diagnosis, reason = classify_failure(
        ground_truth_covered=False,
        goal_sequence_generated=False,
        unpruned_static=0,
        unpruned_executable=0,
        production_executable=0,
        goal_program_rank=None,
        selected_goal_action=False,
        terminal_blocked=False,
        execution_errors=0,
    )
    return ReachabilityRow(
        game=pair.game_id,
        root_key=pair.root_key,
        pair_digest=pair.pair_digest,
        path=pair.path,
        depth=pair.depth,
        ground_truth_covered=False,
        winning_path="",
        remaining_path="",
        required_sequence_length=None,
        requires_multi_action=False,
        goal_action_generated=False,
        goal_sequence_generated=False,
        unpruned_static_goal_programs=0,
        unpruned_executable_goal_programs=0,
        production_executable_goal_programs=0,
        goal_program_rank=None,
        goal_program_mass=0.0,
        goal_action_rank=None,
        goal_action_in_top8=False,
        selected_goal_action=False,
        goal_action_terminal_risk=None,
        goal_sequence_terminal_risk=None,
        diagnosis=diagnosis,
        reason=reason,
    )


def build_report(
    rows: Sequence[ReachabilityRow],
    *,
    pairs: Sequence[BindingPairRecord],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    labeled = [row for row in rows if row.ground_truth_covered]
    unlabeled = [row for row in rows if not row.ground_truth_covered]
    failures = [row for row in labeled if row.diagnosis != "REACHABLE"]
    counts = Counter(row.diagnosis for row in failures)
    by_game = {}
    for game in SOURCE_GAMES:
        game_rows = [row for row in rows if row.game == game]
        game_labeled = [row for row in game_rows if row.ground_truth_covered]
        by_game[game] = {
            "contexts": len(game_rows),
            "ground_truth_covered_contexts": len(game_labeled),
            "diagnoses": dict(Counter(row.diagnosis for row in game_labeled)),
            "selected_goal_actions": sum(row.selected_goal_action for row in game_labeled),
            "mean_goal_action_terminal_risk": (
                mean(
                    float(row.goal_action_terminal_risk)
                    for row in game_labeled
                    if row.goal_action_terminal_risk is not None
                )
                if any(row.goal_action_terminal_risk is not None for row in game_labeled)
                else None
            ),
        }
    primary = counts.most_common(1)[0][0] if counts else "REACHABLE"
    checks = {
        "all_pairs_accounted": len(rows) == len(pairs) == 380,
        "all_arms_accounted": 2 * len(pairs) == 760,
        "three_winning_paths": len({row.root_key for row in labeled}) == 3,
        "nine_winning_prefixes": len(labeled) == 9,
        "exclusive_diagnoses": all(row.diagnosis in DIAGNOSES for row in rows),
        "zero_execution_errors": sum(row.execution_errors for row in rows) == 0,
        "source_train_only": {pair.source_split for pair in pairs} == {"source_train"},
        "firewall_closed": manifest["firewall"]["authority"] == "shadow",
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T9_0_COMPLETE" if all(checks.values()) else "T9_0_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "checks": checks,
        "dataset": {
            "pairs": len(pairs),
            "arms": 2 * len(pairs),
            "roots": len({pair.root_key for pair in pairs}),
            "goal_positive_arms": sum(
                _level_positive(pair, side)
                for pair in pairs
                for side in ("left", "right")
            ),
            "ground_truth_covered_contexts": len(labeled),
            "ground_truth_uncovered_contexts": len(unlabeled),
        },
        "diagnosis": {
            "primary": primary,
            "labeled_failure_counts": dict(counts),
            "reachable_contexts": sum(row.diagnosis == "REACHABLE" for row in labeled),
            "multi_action_contexts": sum(row.requires_multi_action for row in labeled),
            "sequence_generated_contexts": sum(row.goal_sequence_generated for row in labeled),
            "goal_action_top8_contexts": sum(row.goal_action_in_top8 for row in labeled),
            "terminal_blocked_contexts": sum(
                row.diagnosis == "GOAL_CANDIDATE_VETOED" for row in labeled
            ),
        },
        "per_game": by_game,
        "scientific_limit": (
            "su15 has no level-positive arm in the frozen depth-three tree; "
            "its 216 contexts are coverage evidence, not labeled reachability evidence"
        ),
        "t9_1_authorized": all(checks.values()),
        "t9_2_authorized": False,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
        "holdout_opened": False,
    }
    payload["report_checksum"] = v86c._checksum(payload)
    return payload


def run_audit(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, shard_dir=shard_dir)
    pairs = load_pairs(shard_dir, SOURCE_GAMES)
    paths = winner_paths(pairs)
    grouped: dict[str, list[BindingPairRecord]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.root_key].append(pair)
    labeled_rows = []
    labeled_digests = set()
    for root_key, winning_paths in sorted(paths.items()):
        winning = winning_paths[0]
        root_rows = _audit_winning_root(
            grouped[root_key],
            winning_path=winning,
            manifest=manifest,
        )
        labeled_rows.extend(root_rows)
        labeled_digests.update(row.pair_digest for row in root_rows)
    rows = tuple(
        sorted(
            (
                *labeled_rows,
                *(
                    _uncovered_row(pair)
                    for pair in pairs
                    if pair.pair_digest not in labeled_digests
                ),
            ),
            key=lambda row: (row.game, row.root_key, row.depth, row.path),
        )
    )
    report = build_report(rows, pairs=pairs, manifest=manifest)
    destination = Path(output_dir)
    v86c._write_jsonl(destination / "rows.jsonl", (asdict(row) for row in rows))
    v86c._write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(
            output_path=args.manifest,
            shard_dir=args.shard_dir,
        )
    else:
        result = run_audit(
            manifest_path=args.manifest,
            shard_dir=args.shard_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_0_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CURRENT_CAPS",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SHARD_DIR",
    "DIAGNOSES",
    "FORMAT_VERSION",
    "ReachabilityRow",
    "build_report",
    "classify_failure",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_audit",
    "winner_paths",
]
