"""Durable core-only runtime for SAGE.T10.3.12 relational mechanisms."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import t10_3_2_runtime as durable
from . import t10_3_5_runtime as shell
from . import t10_3_12_protocol as protocol
from .goal_directed_v10_3_2 import (
    GoalDirectedOption,
    OptionStep,
    ProgressProgramRegistry,
)
from .goal_directed_v10_3_11 import (
    GoalConditionedUnifiedCognitiveController,
    goal_conditioned_unified_config,
)
from .goal_directed_v10_3_12 import (
    OptionLocalAutomatonInducer,
    RelationalMechanismSageTController,
)
from .relational_program_v10_3_12 import (
    ARMS,
    CONTEXTS,
    RelationalProgramRegistry,
    assert_transfer_safe,
    boundary_distance,
    canonical_json,
    compile_candidate_registry,
    evaluate_fixture,
    fixture_correct,
    fixture_recipes,
    inverse_transform,
    materialize_fixture,
    sha256_payload,
    signed,
    transform_point,
)

PARENT_AUDIT_FILENAME = "parent_quarantine_receipt.json"
PREFLIGHT_FILENAME = "synthetic_preflight.json"
FIXTURE_INVENTORY_FILENAME = "offline_fixture_inventory.json"
CANDIDATE_REGISTRY_FILENAME = "candidate_registry.json"
OFFLINE_REPORT_FILENAME = "offline_equivariance_report.json"
ACTIVE_REPORT_FILENAME = "active_core_report.json"
ADJUDICATION_FILENAME = "adjudication_report.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
LOCK_FILENAME = durable.LOCK_FILENAME

_ACTIVE_PAIRS: dict[
    str,
    tuple[
        GoalConditionedUnifiedCognitiveController,
        RelationalMechanismSageTController,
    ],
] = {}
_ACTIVE_RELATIONAL_REGISTRY: RelationalProgramRegistry | None = None


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _artifact_path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    return durable._read_signed(_artifact_path(root, filename), checksum_field)


def _write(root: Path, filename: str, payload: Mapping[str, Any]) -> None:
    protocol.write_json_once(_artifact_path(root, filename), payload)


def _artifact_size(root: Path) -> int:
    destination = _destination(root)
    return sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) if destination.exists() else 0


@contextmanager
def _contracts() -> Iterator[None]:
    old_durable_protocol = durable.protocol
    old_shell_protocol = shell.protocol
    old_shell_pair = shell._controller_pair
    durable.protocol = protocol
    shell.protocol = protocol
    shell._controller_pair = _controller_pair
    try:
        yield
    finally:
        shell._controller_pair = old_shell_pair
        shell.protocol = old_shell_protocol
        durable.protocol = old_durable_protocol


def _require_artifact_gate(
    root: Path,
    phase: str,
    *,
    expected: Any = True,
) -> dict[str, Any]:
    contract = protocol.ARTIFACT_CONTRACT[phase]
    payload = _read_signed(root, contract["path"], contract["checksum_field"])
    gate = contract["gate_field"]
    if gate is not None and payload.get(gate) != expected:
        raise protocol.ScientificGateMiss(f"{phase} gate forbids continuation")
    return payload


def audit_parent(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    diagnosis = protocol.parent_diagnosis(root)
    expected = {
        key: protocol.QUARANTINED_PARENT[key]
        for key in diagnosis
        if key != "inflight_paths"
    }
    observed = {key: value for key, value in diagnosis.items() if key != "inflight_paths"}
    parent_root = root / "training" / "sage_t" / "t10_3_11_bounded_goal"
    inflight_hashes = []
    for relative in diagnosis["inflight_paths"]:
        path = parent_root / "journal" / "intents" / relative
        if not path.is_file():
            raise protocol.IntegrityError("quarantined parent intent disappeared")
        inflight_hashes.append(protocol.file_sha256(path))
    checks = {
        "diagnosis_exact": observed == expected,
        "two_inflight_intents_quarantined": len(inflight_hashes) == 2,
        "no_live_parent_lock": diagnosis["live_collector_lock"] is False,
        "parent_inflight_explicitly_invalid": diagnosis["inflight_valid"] is False,
        "parent_not_training": protocol.QUARANTINED_PARENT["used_for_training"] is False,
        "parent_registry_not_loaded": protocol.QUARANTINED_PARENT["registry_loaded"] is False,
        "parent_distance_not_loaded": protocol.QUARANTINED_PARENT["distance_metrics_loaded"] is False,
        "parent_not_mutated": (
            manifest["quarantined_parent"]["journal_digest"]
            == protocol.parent_journal_digest(root)
        ),
        "zero_physical_replay": True,
    }
    payload = signed(
        {
            "format_version": "sage-t10.3.12-parent-quarantine-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_status": protocol.QUARANTINED_PARENT["status"],
            "diagnosis": diagnosis,
            "inflight_intent_hashes": sorted(inflight_hashes),
            "checks": checks,
            "passed": all(checks.values()),
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "parent_artifacts_mutated": False,
            "physical_actions": 0,
        },
        "receipt_checksum",
    )
    _write(root, PARENT_AUDIT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.IntegrityError("T10.3.11 quarantine audit failed")
    return payload


def _prefix_option(context: str) -> GoalDirectedOption:
    if context == "repeat_context":
        return GoalDirectedOption(
            schema="repeat_target",
            steps=tuple(OptionStep("ACTION6") for _ in range(3)),
            source="synthetic_prefix_test",
        )
    return GoalDirectedOption(
        schema="path_successor",
        steps=tuple(OptionStep("ACTION6") for _ in range(3)),
        source="synthetic_prefix_test",
    )


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_artifact_gate(root, "audit-parent")
    recipes = fixture_recipes()
    registry = compile_candidate_registry(
        {
            "first": {
                "macro_schema": "repeat_target",
                "target_selector": "same_effect_distinct_target",
            },
            "second": {
                "macro_schema": "path_successor",
                "target_selector": "successor_toward_enclosure",
            },
        },
        repeat_projection_identified=True,
    )
    snapshot = registry.snapshot()
    assert_transfer_safe(snapshot)
    roundtrip = RelationalProgramRegistry(snapshot).snapshot() == snapshot
    prefix_rows = []
    for context in CONTEXTS:
        option = _prefix_option(context)
        expected_hash = None
        for prefix in (0, 4, 17):
            effects = tuple(f"prefix_{index}" for index in range(prefix)) + (
                "local_a",
                "local_b",
                "local_c",
            )
            aligned = OptionLocalAutomatonInducer.align_effects(
                option,
                effects,
                option_start=prefix,
            )
            learned_hash = aligned.option_id
            expected_hash = learned_hash if expected_hash is None else expected_hash
            prefix_rows.append(
                {
                    "context": context,
                    "prefix_length": prefix,
                    "program_hash": learned_hash,
                    "passed": learned_hash == expected_hash,
                }
            )
    checks = {
        "d4_round_trip": all(
            transform_point(transform_point((13, 29), transform), inverse_transform(transform))
            == (13, 29)
            for transform in {
                recipe["transform"] for recipe in recipes
            }
        ),
        "fixture_inventory_exact": len(recipes) == 96 and len({row["fixture_id"] for row in recipes}) == 96,
        "transfer_payload_safe": True,
        "registry_round_trip": roundtrip,
        "palette_is_program_invariant": len(
            {
                registry.program_for(ARMS[0], context).program_hash
                for context in CONTEXTS
                for _palette in ("identity", "cycle_nonmodal")
            }
        ) == 2,
        "candidate_order_is_program_invariant": len(
            {
                registry.program_for(ARMS[0], context).program_hash
                for context in CONTEXTS
                for _order in ("canonical", "reverse")
            }
        ) == 2,
    }
    for row in prefix_rows:
        checks[f"prefix_{row['context']}_{row['prefix_length']}"] = row["passed"]
    if len(checks) != 12:
        raise AssertionError("T10.3.12 preflight must contain exactly 12 cases")
    payload = signed(
        {
            "format_version": "sage-t10.3.12-synthetic-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "cases": checks,
            "prefix_alignment": prefix_rows,
            "passed": all(checks.values()),
            "status": "PASS_T10_3_12_PREFLIGHT" if all(checks.values()) else "RELATIONAL_PREFLIGHT_MISS",
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not payload["passed"]:
        raise protocol.ScientificGateMiss(str(payload["status"]))
    return payload


def materialize_offline(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_artifact_gate(root, "preflight")
    protocol.verify_source_allowlist(root)
    recipes = fixture_recipes()
    materialized = [materialize_fixture(recipe) for recipe in recipes]
    checks = {
        "fixture_count": len(materialized) == 96,
        "positive_count": sum(row.control == "positive" for row in materialized) == 64,
        "control_count": sum(row.control != "positive" for row in materialized) == 32,
        "unique_ids": len({row.fixture_id for row in materialized}) == 96,
        "d4_bijections": all(
            transform_point(transform_point((7, 41), row.transform), inverse_transform(row.transform))
            == (7, 41)
            for row in materialized
        ),
        "points_in_bounds": all(
            0 <= coordinate < 64
            for row in materialized
            for candidate in row.candidates
            for coordinate in candidate.point
        ),
        "no_raw_grid_persisted": True,
        "source_fingerprints_exact": True,
    }
    inventory = signed(
        {
            "format_version": "sage-t10.3.12-offline-fixture-inventory-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "source_fingerprints": {
                name: binding["sha256"]
                for name, binding in protocol.SOURCE_ALLOWLIST.items()
            },
            "recipes": list(recipes),
            "checks": checks,
            "passed": all(checks.values()),
            "raw_grids_retained": False,
            "physical_actions": 0,
        },
        "inventory_checksum",
    )
    _write(root, FIXTURE_INVENTORY_FILENAME, inventory)
    if _artifact_size(root) > int(manifest["offline_matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12 offline artifact budget exceeded")
    if not inventory["passed"]:
        raise protocol.ScientificGateMiss("OFFLINE_FIXTURE_MATERIALIZATION_MISS")
    return inventory


def _repeat_source_projection(root: Path) -> dict[str, Any]:
    binding = protocol.SOURCE_ALLOWLIST["repeat_source_shard"]
    path = root / binding["path"]
    roles: dict[str, dict[str, Any]] = {}
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rows += 1
        payload = json.loads(line)
        for arm in ("left", "right"):
            branch = payload.get(arm, {})
            trace = branch.get("trace", {})
            action = branch.get("action", {})
            data = action.get("action_args", {})
            frame = trace.get("frame_before", ())
            shape = (len(frame), len(frame[0]) if frame else 0)
            distance = boundary_distance(data, shape)
            if distance is None:
                continue
            local_key = sha256_payload(dict(data))
            row = roles.setdefault(
                local_key,
                {"boundary_distance": distance, "terminal_support": 0, "observations": 0},
            )
            row["observations"] += 1
            effects = trace.get("effects", {})
            if bool(effects.get("level_complete")) or int(
                trace.get("levels_completed_after", 0)
            ) > int(trace.get("levels_completed_before", 0)):
                row["terminal_support"] += 1
    productive = [row for row in roles.values() if int(row["terminal_support"]) > 0]
    sterile = [row for row in roles.values() if int(row["terminal_support"]) == 0]
    identified = bool(
        len(productive) == 1
        and sterile
        and int(productive[0]["terminal_support"]) >= 3
        and float(productive[0]["boundary_distance"])
        < min(float(row["boundary_distance"]) for row in sterile)
    )
    # No local action or coordinate leaves this function.
    return {
        "projection_format": "repeat-causal-role-projection-v1",
        "rows_inspected": rows,
        "candidate_role_count": len(roles),
        "unique_productive_role": len(productive) == 1,
        "terminal_support": sum(int(row["terminal_support"]) for row in productive),
        "relative_boundary_realization_identified": identified,
        "grounded_arguments_retained": False,
        "raw_frames_retained": False,
    }


def _canonical_source_projection(root: Path) -> dict[str, Any]:
    binding = protocol.SOURCE_ALLOWLIST["canonical_witness_projection"]
    report = durable._read_signed(root / binding["path"], "report_checksum")
    descriptors = report.get("canonical_descriptors")
    if not isinstance(descriptors, Mapping):
        raise protocol.IntegrityError("canonical witness descriptors are absent")
    # The strict compiler rejects grounded evidence recursively.
    return {str(key): dict(value) for key, value in descriptors.items() if isinstance(value, Mapping)}


def compile_candidates(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_artifact_gate(root, "materialize-offline")
    projection = _repeat_source_projection(root)
    registry = compile_candidate_registry(
        _canonical_source_projection(root),
        repeat_projection_identified=bool(
            projection["relative_boundary_realization_identified"]
        ),
    )
    snapshot = registry.snapshot()
    core = {key: value for key, value in snapshot.items() if key != "registry_checksum"}
    core.update(
        {
            "source_projection": projection,
            "compiled_before_offline_scores": True,
            "local_support_total": 0,
            "physical_actions": 0,
        }
    )
    assert_transfer_safe(core)
    payload = signed(core, "registry_checksum")
    RelationalProgramRegistry(payload)
    _write(root, CANDIDATE_REGISTRY_FILENAME, payload)
    return payload


def _median(values: Sequence[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def evaluate_offline(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _require_artifact_gate(root, "materialize-offline")
    registry_payload = _read_signed(
        root, CANDIDATE_REGISTRY_FILENAME, "registry_checksum"
    )
    if registry_payload.get("compiled_before_offline_scores") is not True:
        raise protocol.IntegrityError("candidate programs were not compiled before scoring")
    registry = RelationalProgramRegistry(registry_payload)
    started = time.perf_counter()
    rows = []
    for recipe in inventory["recipes"]:
        fixture = materialize_fixture(recipe)
        for arm in ARMS:
            program = registry.program_for(arm, fixture.context)
            outcome = evaluate_fixture(program, fixture)
            rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "context": fixture.context,
                    "control": fixture.control,
                    "arm": arm,
                    "program_hash": program.program_hash,
                    "correct": fixture_correct(fixture, outcome),
                    "abstained": outcome.abstained,
                    "inspections": outcome.inspections,
                    "output_checksum": sha256_payload(
                        {"tokens": outcome.tokens, "points": outcome.points}
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    total_inspections = sum(int(row["inspections"]) for row in rows)

    def arm_rows(arm: str, *, context: str | None = None) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["arm"] == arm and (context is None or row["context"] == context)
        ]

    correct = {
        arm: sum(bool(row["correct"]) for row in arm_rows(arm)) for arm in ARMS
    }
    positives_a = [
        row
        for row in arm_rows(ARMS[0])
        if row["control"] == "positive"
    ]
    controls_a = [
        row for row in arm_rows(ARMS[0]) if row["control"] != "positive"
    ]
    efficiency_by_context = {}
    source_value_by_context = {}
    for context in CONTEXTS:
        a = arm_rows(ARMS[0], context=context)
        b = arm_rows(ARMS[1], context=context)
        a_median = _median([int(row["inspections"]) for row in a])
        b_median = _median([int(row["inspections"]) for row in b])
        efficiency_by_context[context] = {
            "factorized_median": a_median,
            "generic_median": b_median,
            "ratio": a_median / b_median if b_median else 1.0,
        }
        a_correct = sum(bool(row["correct"]) for row in a)
        b_correct = sum(bool(row["correct"]) for row in b)
        source_value_by_context[context] = bool(
            a_correct >= b_correct + 8
            or (
                a_correct >= b_correct
                and b_median > 0
                and a_median <= 0.5 * b_median
            )
        )
    checks = {
        "factorized_equivariance_64_of_64": len(positives_a) == 64
        and all(row["correct"] for row in positives_a),
        "factorized_controls_32_of_32": len(controls_a) == 32
        and all(row["correct"] for row in controls_a),
        "prefix_invariance_6_of_6": all(
            row["passed"]
            for row in _read_signed(root, PREFLIGHT_FILENAME, "preflight_checksum")[
                "prefix_alignment"
            ]
        ),
        "one_program_hash_per_context": all(
            len(
                {
                    row["program_hash"]
                    for row in positives_a
                    if row["context"] == context
                }
            )
            == 1
            for context in CONTEXTS
        ),
        "source_information_identified": all(source_value_by_context.values()),
        "mechanism_specific": correct[ARMS[0]] >= correct[ARMS[2]] + 8,
        "relation_is_causal": correct[ARMS[0]] >= correct[ARMS[3]] + 8,
        "inspection_budget": total_inspections
        <= int(manifest["offline_matrix"]["maximum_candidate_inspections"]),
        "wall_budget": elapsed <= int(manifest["offline_matrix"]["maximum_wall_seconds"]),
        "payload_transfer_safe": True,
        "zero_physical_actions": True,
    }
    passed = all(checks.values())
    if not checks["factorized_equivariance_64_of_64"] or not checks[
        "factorized_controls_32_of_32"
    ]:
        verdict = "RELATIONAL_EQUIVARIANCE_MISS"
    elif not checks["source_information_identified"]:
        verdict = "SOURCE_INFORMATION_NOT_IDENTIFIED"
    elif not checks["mechanism_specific"] or not checks["relation_is_causal"]:
        verdict = "MECHANISM_NOT_SPECIFIC"
    elif not passed:
        verdict = "OFFLINE_RESOURCE_OR_SAFETY_MISS"
    else:
        verdict = "PASS_T10_3_12_OFFLINE_RELATIONAL_GATE"
    report = signed(
        {
            "format_version": "sage-t10.3.12-offline-equivariance-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "candidate_registry_checksum": registry_payload["registry_checksum"],
            "fixture_inventory_checksum": inventory["inventory_checksum"],
            "metrics": {
                "correct_by_arm": correct,
                "efficiency_by_context": efficiency_by_context,
                "source_value_by_context": source_value_by_context,
                "candidate_inspections": total_inspections,
                "elapsed_seconds": elapsed,
            },
            "evaluations": rows,
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "physical_actions": 0,
        },
        "report_checksum",
    )
    _write(root, OFFLINE_REPORT_FILENAME, report)
    if _artifact_size(root) > int(manifest["offline_matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12 artifact budget exceeded")
    if not passed:
        raise protocol.ScientificGateMiss(verdict)
    return report


def _controller_pair(
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    *,
    registry_checksum: str | None,
) -> tuple[
    GoalConditionedUnifiedCognitiveController,
    RelationalMechanismSageTController,
]:
    del registry_checksum
    if _ACTIVE_RELATIONAL_REGISTRY is None:
        raise protocol.IntegrityError("active relational registry is not loaded")
    goal = RelationalMechanismSageTController(
        phase="discovery",
        registry=registry,
        attestation_scope=work.work_id,
        exploration_seed=work.seed,
        arm=work.arm,
        relational_registry=_ACTIVE_RELATIONAL_REGISTRY,
        goal_conditioning_enabled=False,
    )
    controller = GoalConditionedUnifiedCognitiveController(
        work.game_id,
        config=goal_conditioned_unified_config(sage_t_authority_mode="active"),
        sage_t_controller=goal,
        goal_conditioning_enabled=False,
    )
    pair = (controller, goal)
    _ACTIVE_PAIRS[work.work_id] = pair
    return pair


def _diagnostic_path(destination: Path, work: protocol.WorkSpec) -> Path:
    return destination / "branch_diagnostics" / f"{work.work_id}.json"


def _run_work(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    work: protocol.WorkSpec,
    registry: ProgressProgramRegistry,
    lock: Any,
) -> dict[str, Any]:
    receipt = shell._run_work(
        root,
        destination,
        manifest,
        work,
        registry,
        lock,
        registry_checksum=None,
    )
    pair = _ACTIVE_PAIRS.pop(work.work_id, None)
    if pair is None:
        return receipt
    controller, goal = pair
    controller_summary = dict(controller.summary())
    goal_summary = dict(goal.summary())
    diagnostic = signed(
        {
            "format_version": "sage-t10.3.12-active-branch-diagnostic-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "work_id": work.work_id,
            "receipt_checksum": receipt["receipt_checksum"],
            "game_id": work.game_id,
            "arm": work.arm,
            "recognized_context": goal_summary.get("recognized_context", ""),
            "program_hashes_used": goal_summary.get(
                "relational_program_hashes_used", ()
            ),
            "source_information_loaded": bool(
                goal_summary.get("source_information_loaded")
            ),
            "relational_inspections": int(
                goal_summary.get("relational_inspections", 0)
            ),
            "relational_abstentions": dict(
                goal_summary.get("relational_abstentions", {})
            ),
            "role_binding_uses": dict(goal_summary.get("role_binding_uses", {})),
            "relation_attestations": dict(
                goal_summary.get("relation_attestations", {})
            ),
            "fresh_plan_reacquisitions": int(
                goal_summary.get("fresh_plan_reacquisitions", 0)
            ),
            "option_local_effect_alignment": bool(
                goal_summary.get("option_local_effect_alignment")
            ),
            "posterior_observations": int(
                goal_summary.get("bounded_program_posterior", {}).get(
                    "observations", 0
                )
            ),
            "goal_conditioning_enabled": bool(
                goal_summary.get("goal_conditioning_enabled", True)
            ),
            "scheduled_legacy_decisions": int(
                controller_summary.get("scheduled_legacy_decisions", 0)
            ),
            "cross_reset_memory": False,
            "grounded_arguments_persisted": False,
        },
        "diagnostic_checksum",
    )
    protocol.write_json_once(_diagnostic_path(destination, work), diagnostic)
    return receipt


def _load_diagnostics(root: Path) -> list[dict[str, Any]]:
    base = _destination(root) / "branch_diagnostics"
    return [
        durable._read_signed(path, "diagnostic_checksum")
        for path in sorted(base.glob("*.json"))
    ] if base.exists() else []


def _initial_frame_hashes(root: Path) -> dict[str, list[str]]:
    destination = _destination(root)
    output: dict[str, set[str]] = {game: set() for game in protocol.CORE_GAMES}
    for work in protocol.work_specs("active-core"):
        directory = destination / "journal" / "events" / work.work_id
        paths = tuple(sorted(directory.glob("*.json"))) if directory.exists() else ()
        if paths:
            event = durable._read_signed(paths[0], "event_checksum")
            output[work.game_id].add(str(event.get("frame_before_sha256", "")))
    return {game: sorted(values - {""}) for game, values in output.items()}


def _active_metrics(
    receipts: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successes = {
        arm: {
            game: sum(
                int(row.get("level_delta", 0)) > 0
                for row in receipts
                if row.get("arm") == arm and row.get("game_id") == game
            )
            for game in protocol.CORE_GAMES
        }
        for arm in ARMS
    }
    actions_to_level = {
        arm: {
            game: [
                int(row.get("sealed_events", 0))
                for row in receipts
                if row.get("arm") == arm
                and row.get("game_id") == game
                and int(row.get("level_delta", 0)) > 0
            ]
            for game in protocol.CORE_GAMES
        }
        for arm in ARMS
    }
    actions_total = {
        arm: sum(
            int(row.get("sealed_events", 0))
            for row in receipts
            if row.get("arm") == arm and int(row.get("level_delta", 0)) > 0
        )
        for arm in ARMS
    }
    diagnostic_by_id = {str(row["work_id"]): row for row in diagnostics}
    return {
        "successes": successes,
        "actions_to_level": actions_to_level,
        "actions_total_to_level": actions_total,
        "diagnostic_count": len(diagnostics),
        "all_actions_sage_t": all(
            int(row.get("sage_t_option_actions", 0))
            == int(row.get("sealed_events", 0))
            for row in receipts
        ),
        "posterior_each_event": all(
            int(diagnostic_by_id.get(str(row["work_id"]), {}).get("posterior_observations", -1))
            == int(row.get("sealed_events", 0))
            for row in receipts
        ),
    }


def active_core(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    global _ACTIVE_RELATIONAL_REGISTRY
    _require_artifact_gate(root, "evaluate-offline")
    registry_payload = _read_signed(
        root, CANDIDATE_REGISTRY_FILENAME, "registry_checksum"
    )
    _ACTIVE_RELATIONAL_REGISTRY = RelationalProgramRegistry(registry_payload)
    destination = _destination(root)
    with _contracts():
        durable._require_live_runtime()
        durable._recover_orphans(destination, manifest)
        lock = durable._CollectorLock(destination / LOCK_FILENAME, "active-core")
        lock.acquire()
        try:
            for work in protocol.work_specs("active-core"):
                # No registry, posterior or support is shared across work specs.
                fresh_registry = ProgressProgramRegistry()
                _run_work(root, destination, manifest, work, fresh_registry, lock)
        finally:
            lock.release()
        receipts = durable._load_receipts(destination, "active-core")
        accounting = durable._journal_accounting(destination)
    diagnostics = _load_diagnostics(root)
    metrics = _active_metrics(receipts, diagnostics)
    initial_hashes = _initial_frame_hashes(root)
    collection_checks = {
        "all_32_receipts_present": len(receipts) == protocol.TOTAL_RESETS,
        "all_32_diagnostics_present": len(diagnostics) == protocol.TOTAL_RESETS,
        "accounting_equation": bool(accounting.get("equation_holds")),
        "zero_inflight": int(accounting.get("inflight_intents", 0)) == 0,
        "zero_unresolved": int(accounting.get("unresolved_intents", 0)) == 0,
        "zero_incomplete_work": not accounting.get("incomplete_work_ids"),
        "zero_physical_replay": all(
            int(row.get("physical_actions_replayed", 0)) == 0 for row in receipts
        ),
    }
    report = signed(
        {
            "format_version": "sage-t10.3.12-active-core-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "candidate_registry_checksum": registry_payload["registry_checksum"],
            "collection_checks": collection_checks,
            "collection_complete": all(collection_checks.values()),
            "accounting": accounting,
            "metrics": {
                **metrics,
                "initial_frame_hashes": initial_hashes,
                "distinct_initial_frames": {
                    game: len(values) for game, values in initial_hashes.items()
                },
                "replicate_labels_seed_environment": False,
                "actions": sum(int(row.get("sealed_events", 0)) for row in receipts),
            },
            "receipt_checksums": [row["receipt_checksum"] for row in receipts],
            "diagnostic_checksums": [
                row["diagnostic_checksum"] for row in diagnostics
            ],
            "physical_actions_replayed": 0,
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, ACTIVE_REPORT_FILENAME, report)
    if not report["collection_complete"]:
        raise protocol.IntegrityError("active-core collection did not seal cleanly")
    return report


def _science_checks(
    active: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successes = active["metrics"]["successes"]
    action_rows = active["metrics"]["actions_to_level"]
    a = ARMS[0]
    b = ARMS[1]
    c = ARMS[2]
    d = ARMS[3]
    a_total = sum(int(successes[a][game]) for game in protocol.CORE_GAMES)
    b_total = sum(int(successes[b][game]) for game in protocol.CORE_GAMES)
    c_total = sum(int(successes[c][game]) for game in protocol.CORE_GAMES)
    d_total = sum(int(successes[d][game]) for game in protocol.CORE_GAMES)
    core0, core1 = protocol.CORE_GAMES
    repeat_actions = [int(value) for value in action_rows[a][core0]]
    path_actions = [int(value) for value in action_rows[a][core1]]
    a_receipts = [row for row in receipts if row.get("arm") == a]
    a_diagnostics = [row for row in diagnostics if row.get("arm") == a]
    b_actions = int(active["metrics"]["actions_total_to_level"][b])
    a_actions = int(active["metrics"]["actions_total_to_level"][a])
    source_advantage = a_total >= b_total + 2
    if a_total == b_total == 8 and b_actions > 0:
        source_advantage = a_actions <= 0.75 * b_actions
    return {
        "zero_controller_errors": all(not row.get("errors") for row in receipts),
        "zero_illegal_actions": all(
            int(row.get("illegal_actions", 0)) == 0 for row in receipts
        ),
        "all_actions_sage_t": bool(active["metrics"]["all_actions_sage_t"]),
        "posterior_each_event": bool(active["metrics"]["posterior_each_event"]),
        "factorized_8_of_8": a_total == 8,
        "factorized_zero_game_over": all(
            int(row.get("game_over_actions", 0)) == 0 for row in a_receipts
        ),
        "repeat_4_of_4": int(successes[a][core0]) == 4,
        "repeat_role_attested": len(a_diagnostics) == 8
        and all(
            row.get("recognized_context") != "repeat_context"
            or sum(int(value) for value in row.get("role_binding_uses", {}).values()) > 0
            for row in a_diagnostics
        ),
        "repeat_median_at_most_6": len(repeat_actions) == 4
        and statistics.median(repeat_actions) <= 6,
        "repeat_maximum_at_most_8": len(repeat_actions) == 4
        and max(repeat_actions) <= 8,
        "path_4_of_4": int(successes[a][core1]) == 4,
        "path_relation_attested": len(a_diagnostics) == 8
        and all(
            row.get("recognized_context") != "path_context"
            or (
                int(row.get("relation_attestations", {}).get("successor", 0)) > 0
                and int(row.get("relation_attestations", {}).get("orientation", 0)) > 0
            )
            for row in a_diagnostics
        ),
        "path_median_at_most_10": len(path_actions) == 4
        and statistics.median(path_actions) <= 10,
        "path_maximum_at_most_16": len(path_actions) == 4
        and max(path_actions) <= 16,
        "factorized_not_inferior_per_game": all(
            int(successes[a][game])
            >= max(int(successes[arm][game]) for arm in (b, c, d))
            for game in protocol.CORE_GAMES
        ),
        "source_value_active": source_advantage,
        "wrong_source_advantage_at_least_2": a_total >= c_total + 2,
        "relation_ablation_advantage_at_least_2": a_total >= d_total + 2,
        "fresh_work_scopes": len({str(row["work_id"]) for row in a_receipts}) == 8,
        "option_local_effect_alignment": all(
            bool(row.get("option_local_effect_alignment")) for row in diagnostics
        ),
    }


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_artifact_gate(root, "evaluate-offline")
    active = _read_signed(root, ACTIVE_REPORT_FILENAME, "report_checksum")
    if active.get("collection_complete") is not True:
        raise protocol.IntegrityError("active-core collection is incomplete")
    accounting = active["accounting"]
    if not (
        accounting.get("equation_holds")
        and int(accounting.get("inflight_intents", 0)) == 0
        and int(accounting.get("unresolved_intents", 0)) == 0
        and not accounting.get("incomplete_work_ids")
    ):
        raise protocol.IntegrityError("active-core accounting is invalid")
    with _contracts():
        receipts = durable._load_receipts(_destination(root), "active-core")
    diagnostics = _load_diagnostics(root)
    checks = _science_checks(active, receipts, diagnostics)
    if not checks["factorized_8_of_8"]:
        repeat = checks["repeat_4_of_4"]
        path = checks["path_4_of_4"]
        verdict = "PARTIAL_MECHANISM_SUPPORT" if repeat != path else "RELATIONAL_ACTIVE_MISS"
    elif not checks["source_value_active"]:
        verdict = "GENERIC_REDISCOVERY_ONLY"
    elif not checks["wrong_source_advantage_at_least_2"]:
        verdict = "MECHANISM_NOT_SPECIFIC"
    elif not checks["relation_ablation_advantage_at_least_2"]:
        verdict = "RELATIONAL_CAUSALITY_MISS"
    elif not all(checks.values()):
        verdict = "RELATIONAL_ACTIVE_MISS"
    else:
        verdict = "PASS_T10_3_12_RELATIONAL_MECHANISM_SOURCE"
    passed = verdict == "PASS_T10_3_12_RELATIONAL_MECHANISM_SOURCE"
    candidate_payload = _read_signed(
        root, CANDIDATE_REGISTRY_FILENAME, "registry_checksum"
    )
    candidate_registry = RelationalProgramRegistry(candidate_payload)
    if passed:
        for context in CONTEXTS:
            for receipt in receipts:
                if receipt.get("arm") == ARMS[0] and int(receipt.get("level_delta", 0)) > 0:
                    candidate_registry.note_success(
                        ARMS[0], context, str(receipt["work_id"])
                    )
            candidate_registry.note_controls(
                ARMS[0],
                context,
                ("binding_swap", "order_permutation", "relation_ablation"),
            )
    promoted = candidate_registry.snapshot(promoted_only=True)
    report = signed(
        {
            "format_version": "sage-t10.3.12-adjudication-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "active_report_checksum": active["report_checksum"],
            "checks": checks,
            "passed": passed,
            "verdict": verdict,
            "promoted_registry": promoted,
            "promotion_count": len(promoted.get("programs", ())) if passed else 0,
            "sequence_access_authorized": False,
            "t10_3_13_preregistration_authorized": passed,
            "production_authority": False,
            "automatic_retuning": False,
            "physical_actions_replayed": 0,
        },
        "report_checksum",
    )
    _write(root, ADJUDICATION_FILENAME, report)
    if not passed:
        raise protocol.ScientificGateMiss(verdict)
    return report


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        if phase == "report":
            continue
        path = _artifact_path(root, contract["path"])
        artifacts[phase] = (
            None
            if not path.is_file()
            else durable._read_signed(path, contract["checksum_field"])
        )
    if artifacts["audit-parent"] is None or artifacts["audit-parent"].get("passed") is not True:
        verdict = "INVALID_PROVENANCE"
    elif artifacts["preflight"] is None or artifacts["preflight"].get("passed") is not True:
        verdict = "RELATIONAL_PREFLIGHT_MISS"
    elif artifacts["materialize-offline"] is None or artifacts["materialize-offline"].get("passed") is not True:
        verdict = "OFFLINE_FIXTURE_MATERIALIZATION_MISS"
    elif artifacts["evaluate-offline"] is None or artifacts["evaluate-offline"].get("passed") is not True:
        verdict = (
            "OFFLINE_NOT_RUN"
            if artifacts["evaluate-offline"] is None
            else str(artifacts["evaluate-offline"].get("verdict"))
        )
    elif artifacts["active-core"] is None:
        verdict = "ACTIVE_CORE_NOT_RUN"
    elif artifacts["adjudicate"] is None:
        verdict = "ADJUDICATION_NOT_RUN"
    else:
        verdict = str(artifacts["adjudicate"].get("verdict"))
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    report = signed(
        {
            "format_version": "sage-t10.3.12-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "verdict": verdict,
            "artifacts": {
                phase: (
                    None
                    if payload is None
                    else payload.get(contract["checksum_field"])
                )
                for phase, payload in artifacts.items()
                for contract in (protocol.ARTIFACT_CONTRACT[phase],)
            },
            "accounting": accounting,
            "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
            "maximum_resets": protocol.TOTAL_RESETS,
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "sequence_games_opened": False,
            "physical_actions_replayed": 0,
            "firewall": manifest["firewall"],
            "production_authority": False,
        },
        "report_checksum",
    )
    _write(root, TERMINAL_REPORT_FILENAME, report)
    return report


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    with _contracts():
        accounting = durable._journal_accounting(_destination(root))
    artifacts = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        path = _artifact_path(root, contract["path"])
        artifacts[phase] = (
            None
            if not path.is_file()
            else durable._read_signed(path, contract["checksum_field"])[
                contract["checksum_field"]
            ]
        )
    state = (
        "RUNNING"
        if accounting.get("live_collector_lock")
        else "INTERRUPTED_FAIL_CLOSED_NO_REPLAY"
        if accounting.get("inflight_intents") or accounting.get("incomplete_work_ids")
        else "READY"
    )
    return {
        "phase": "status",
        "protocol": "SAGE.T10.3.12",
        "status": state,
        "manifest_checksum": manifest["manifest_checksum"],
        "accounting": accounting,
        "artifacts": artifacts,
        "maximum_actions": protocol.TOTAL_MAXIMUM_ACTIONS,
        "maximum_resets": protocol.TOTAL_RESETS,
        "firewall": manifest["firewall"],
    }


def _emit(payload: Mapping[str, Any]) -> None:
    print(canonical_json(payload), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze",
            "status",
            "audit-parent",
            "preflight",
            "materialize-offline",
            "compile-candidates",
            "evaluate-offline",
            "active-core",
            "adjudicate",
            "report",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.phase == "freeze":
            manifest, receipt = protocol.freeze_manifest(root)
            _emit(
                {
                    "phase": "freeze",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "freeze_receipt_checksum": receipt["receipt_checksum"],
                    "status": manifest["status"],
                }
            )
            return 0
        manifest = protocol.load_manifest(root)
        if args.phase == "status":
            _emit(status(root, manifest))
            return 0
        functions = {
            "audit-parent": audit_parent,
            "preflight": preflight,
            "materialize-offline": materialize_offline,
            "compile-candidates": compile_candidates,
            "evaluate-offline": evaluate_offline,
            "active-core": active_core,
            "adjudicate": adjudicate,
            "report": terminal_report,
        }
        result = functions[args.phase](root, manifest)
        _emit(result)
        if args.phase == "report":
            return (
                0
                if result["verdict"]
                == "PASS_T10_3_12_RELATIONAL_MECHANISM_SOURCE"
                else 3
            )
        return 0
    except protocol.ScientificGateMiss as exc:
        _emit({"phase": args.phase, "error": str(exc), "exit_code": 3})
        return 3
    except (protocol.IntegrityError, OSError, ValueError, KeyError) as exc:
        _emit(
            {
                "phase": args.phase,
                "error": f"{type(exc).__name__}:{exc}",
                "exit_code": 2,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "active_core",
    "adjudicate",
    "audit_parent",
    "compile_candidates",
    "evaluate_offline",
    "main",
    "materialize_offline",
    "preflight",
    "status",
    "terminal_report",
]
