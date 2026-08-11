"""Preregistered causal-procedure transfer diagnostic for SAGE T10.3.12f.

T10.3.12f is deliberately retrospective.  It may identify a candidate procedure
for a later, separately authorized T10.3.13 confirmation, but it cannot open or
make claims about the protected holdout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_8_protocol as source_success
from . import t10_3_12e_protocol as parent

FORMAT_VERSION = "sage-t10.3.12f-causal-procedure-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_12F_HISTORICAL_RESET"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_12f_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name(
    "sage_t10_3_12f_freeze_receipt.json"
)
DEFAULT_OUTPUT_DIR = Path("training/sage_t/t10_3_12f_causal_procedure")

PARENT_OUTPUT_DIR = Path("training/sage_t/t10_3_12e_closed_loop_successor")
PARENT_ARTIFACT_PATHS = {
    "manifest": Path("theory/sage_t/sage_t10_3_12e_protocol_manifest.json"),
    "freeze_receipt": Path("theory/sage_t/sage_t10_3_12e_freeze_receipt.json"),
    "audit_parent": PARENT_OUTPUT_DIR / "parent_stable_executor_negative_audit.json",
    "audit_trajectories": (
        PARENT_OUTPUT_DIR / "parent_closed_loop_motivation_audit.json"
    ),
    "preflight": PARENT_OUTPUT_DIR / "closed_loop_preflight.json",
    "registry": PARENT_OUTPUT_DIR / "closed_loop_registry.json",
    "active": PARENT_OUTPUT_DIR / "active_closed_loop_diagnostic.json",
    "adjudication": PARENT_OUTPUT_DIR / "closed_loop_adjudication_report.json",
    "terminal": PARENT_OUTPUT_DIR / "terminal_report.json",
}
EXPECTED_PARENT = {
    "manifest_checksum": (
        "3e7dc757e9c54b8bd0ef0c551060869ad035f3e2fc997ea559f0163177ec451d"
    ),
    "registry_checksum": (
        "16ae0de1fdd9d26149c11ff52703272ffaaff30507323bb10495430d4f050882"
    ),
    "active_report_checksum": (
        "f091810264ba939dd9920fb1a9b4cbbd452aba5cdae146956b61aee5dd020f7f"
    ),
    "adjudication_report_checksum": (
        "9e5f3fbca461fdece673d7150982368d113ed5efb6c06d57096005f5f004a071"
    ),
    "terminal_report_checksum": (
        "e88ed27949bd24221f67952960756558798fe51fbc5085298f0654dbb46e867c"
    ),
    "verdict": "CLOSED_LOOP_NO_PROGRESS",
    "authorized_actions": 82,
    "sealed_events": 82,
    "inflight_intents": 0,
    "unresolved_intents": 0,
    "receipt_count": 36,
    "distinct_target_initial_frames": 9,
    "closed_loop_mechanism_recovered": False,
}

SOURCE_COLLECTION_MANIFEST = Path(
    "training/sage12/bound_mechanic_pilot_v4_3/source_train_collection_manifest.json"
)
SOURCE_SHARD_DIR = Path("training/sage12/bound_mechanic_pilot_v4_3/source_train_shards")
SOURCE_SHARD_SHA256 = {
    "lp85": "7dee5fa89bace32af5f744c02489a22d4703b96e4dea3fef6d6971c9e1f7461c",
    "su15": "d089568302893ba8e409694919d9c744400d53565d3e6f90ddbd5f1012753224",
}
SOURCE_SUCCESS_OUTPUT_DIR = Path("training/sage_t/t10_3_8_witness_gate_adjudication")
SOURCE_SUCCESS_JOURNAL_DIR = SOURCE_SUCCESS_OUTPUT_DIR / "journal"
SOURCE_SUCCESS_JOURNAL_SHA256 = (
    "8ceb4a8a9f82f62b9c023d4862ea5492342ec80dcb96187bc1a3fd9c1aff7efb"
)
SOURCE_SUCCESS_ARTIFACT_PATHS = {
    "manifest": Path("theory/sage_t/sage_t10_3_8_protocol_manifest.json"),
    "migration_receipt": Path("theory/sage_t/sage_t10_3_8_migration_receipt.json"),
    "canonical_witness": SOURCE_SUCCESS_OUTPUT_DIR / "canonical_witness_report.json",
    "discovery_core": SOURCE_SUCCESS_OUTPUT_DIR / "discovery_core_report.json",
    "reproduction_core": SOURCE_SUCCESS_OUTPUT_DIR / "reproduction_core_report.json",
    "terminal": SOURCE_SUCCESS_OUTPUT_DIR / "terminal_report.json",
}
EXPECTED_SOURCE_SUCCESS = {
    "manifest_checksum": (
        "1022e930fcb864bec21715ebc4a3b8049a122c05c9b754c259817043e8abdffc"
    ),
    "canonical_report_checksum": (
        "14266d9ca3f2031f76df89706fc254c06daf34c9a9796f964c977d38de0e1c89"
    ),
    "discovery_report_checksum": (
        "689e4ab908969f6c7d4ce47b59cc0c63403344758eae0c784e9c23e6dbc1bc04"
    ),
    "reproduction_report_checksum": (
        "2591542e386cfa1b65218252440caee7d6316cc540e2e89de15c903923c25dff"
    ),
    "journal_digest": SOURCE_SUCCESS_JOURNAL_SHA256,
    "canonical_levels": {"lp85-305b61c3": 1, "su15-4c352900": 2},
    "discovery_levels": {"lp85-305b61c3": 2, "su15-4c352900": 4},
    "reproduction_levels": {"lp85-305b61c3": 1, "su15-4c352900": 2},
}

TARGET_GAMES = (
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "dc22-4c9bff3e",
    "g50t-5849a774",
    "ka59-9f096b4a",
    "lf52-271a04aa",
    "sp80-0ee2d095",
    "tr87-cd924810",
    "tu93-2b534c15",
)
SOURCE_GAMES = ("lp85-305b61c3", "su15-4c352900")
ARMS = (
    "source_closed_loop",
    "uniform_closed_loop",
    "permuted_source_closed_loop",
    "source_open_loop",
)
MODEL_FAMILIES = (
    "stable_repeat",
    "relational_successor",
    "state_conditioned_switch",
    "null_or_unsafe",
)
WORK_SCOPES = (0, 1, 2, 3)
ACTION_BUDGET = 48
RESET_WALL_SECONDS = 180.0
GLOBAL_WALL_SECONDS = 8 * 60 * 60
MAXIMUM_ARTIFACT_BYTES = 128 * 1024 * 1024
TOTAL_RESETS = len(TARGET_GAMES) * len(ARMS) * len(WORK_SCOPES)
TOTAL_MAXIMUM_ACTIONS = TOTAL_RESETS * ACTION_BUDGET

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
sha256_payload = parent.sha256_payload
write_json_once = parent.write_json_once

ARTIFACT_CONTRACT = {
    "audit": {
        "path": "parent_closed_loop_negative_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "qa-source": {
        "path": "source_causal_qa_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "compile-prior": {
        "path": "causal_procedure_prior.json",
        "checksum_field": "prior_checksum",
        "gate_field": None,
        "role": "registry",
    },
    "evaluate-source": {
        "path": "source_procedure_evaluation.json",
        "checksum_field": "report_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "preflight": {
        "path": "causal_procedure_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "active-historical": {
        "path": "active_historical_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "collection_complete",
        "role": "report",
    },
    "adjudicate": {
        "path": "causal_procedure_adjudication.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
        "role": "report",
    },
    "report": {
        "path": "terminal_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
        "role": "report",
    },
}


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise IntegrityError(f"invalid {checksum_field}")


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _read_signed(root: Path, relative: Path, checksum_field: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise IntegrityError(f"required signed artifact is absent: {relative.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def verify_parent(root: Path) -> dict[str, Any]:
    """Verify the exact clean T10.3.12e negative used to motivate this test."""

    manifest = parent.load_manifest(root)
    for name in ("audit_parent", "audit_trajectories"):
        report = _read_signed(root, PARENT_ARTIFACT_PATHS[name], "audit_checksum")
        if report.get("passed") is not True:
            raise IntegrityError(f"T10.3.12e {name} gate is not a pass")
    preflight = _read_signed(root, PARENT_ARTIFACT_PATHS["preflight"], "preflight_checksum")
    if preflight.get("passed") is not True:
        raise IntegrityError("T10.3.12e preflight gate is not a pass")
    registry = _read_signed(root, PARENT_ARTIFACT_PATHS["registry"], "registry_checksum")
    active = _read_signed(root, PARENT_ARTIFACT_PATHS["active"], "report_checksum")
    adjudication = _read_signed(
        root, PARENT_ARTIFACT_PATHS["adjudication"], "report_checksum"
    )
    terminal = _read_signed(root, PARENT_ARTIFACT_PATHS["terminal"], "report_checksum")
    accounting = terminal.get("accounting", {})
    observed = {
        "manifest_checksum": manifest.get("manifest_checksum"),
        "registry_checksum": registry.get("registry_checksum"),
        "active_report_checksum": active.get("report_checksum"),
        "adjudication_report_checksum": adjudication.get("report_checksum"),
        "terminal_report_checksum": terminal.get("report_checksum"),
        "verdict": terminal.get("verdict"),
        "authorized_actions": accounting.get("authorized_actions"),
        "sealed_events": accounting.get("sealed_events"),
        "inflight_intents": accounting.get("inflight_intents"),
        "unresolved_intents": accounting.get("unresolved_intents"),
        "receipt_count": len(active.get("receipt_checksums", ())),
        "distinct_target_initial_frames": active.get("metrics", {}).get(
            "distinct_target_initial_frames"
        ),
        "closed_loop_mechanism_recovered": terminal.get(
            "closed_loop_mechanism_recovered"
        ),
    }
    if observed != EXPECTED_PARENT:
        raise IntegrityError("T10.3.12e parent state diverged from the frozen negative")
    if not accounting.get("equation_holds") or not accounting.get("inflight_valid"):
        raise IntegrityError("T10.3.12e accounting is not clean")
    if accounting.get("incomplete_work_ids") or accounting.get("inflight_paths"):
        raise IntegrityError("T10.3.12e has incomplete durable work")
    if active.get("collection_complete") is not True:
        raise IntegrityError("T10.3.12e active collection is incomplete")
    if terminal.get("passed") is not False or terminal.get("program_promoted") is not False:
        raise IntegrityError("T10.3.12e terminal polarity drifted")
    for field in (
        "ar25_opened",
        "holdout_opened",
        "new_games_opened",
        "production_authority",
        "sequence_games_opened",
        "source_validation_opened",
    ):
        if terminal.get(field) is not False:
            raise IntegrityError(f"T10.3.12e firewall drifted: {field}")
    if terminal.get("legacy_fallback_actions") != 0:
        raise IntegrityError("T10.3.12e used legacy fallback actions")
    return observed


def parent_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, relative in PARENT_ARTIFACT_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise IntegrityError(f"required T10.3.12e artifact is absent: {name}")
        bindings[name] = {"path": relative.as_posix(), "sha256": file_sha256(path)}
    return bindings


def parent_journal_digest(root: Path) -> str:
    base = root / PARENT_OUTPUT_DIR / "journal"
    rows = [
        {"relative_path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    if not rows:
        raise IntegrityError("T10.3.12e parent journal is absent")
    return sha256_payload(rows)


def verify_source_evidence(root: Path) -> dict[str, Any]:
    """Verify source interventions and successful T10.3.8 endpoints."""

    source_manifest = source_success.load_manifest(root)
    canonical = _read_signed(
        root, SOURCE_SUCCESS_ARTIFACT_PATHS["canonical_witness"], "report_checksum"
    )
    discovery = _read_signed(
        root, SOURCE_SUCCESS_ARTIFACT_PATHS["discovery_core"], "report_checksum"
    )
    reproduction = _read_signed(
        root, SOURCE_SUCCESS_ARTIFACT_PATHS["reproduction_core"], "report_checksum"
    )
    terminal = _read_signed(
        root, SOURCE_SUCCESS_ARTIFACT_PATHS["terminal"], "report_checksum"
    )
    observed = {
        "manifest_checksum": source_manifest.get("manifest_checksum"),
        "canonical_report_checksum": canonical.get("report_checksum"),
        "discovery_report_checksum": discovery.get("report_checksum"),
        "reproduction_report_checksum": reproduction.get("report_checksum"),
        "journal_digest": source_success_journal_digest(root),
        "canonical_levels": canonical.get("metrics", {}).get("levels"),
        "discovery_levels": discovery.get("metrics", {}).get("levels"),
        "reproduction_levels": reproduction.get("metrics", {}).get("levels"),
    }
    if observed != EXPECTED_SOURCE_SUCCESS:
        raise IntegrityError("T10.3.8 lp85/su15 success evidence drifted")
    for report in (canonical, discovery, reproduction):
        if report.get("passed") is not True:
            raise IntegrityError("a bound T10.3.8 source-success report is not a pass")
    if canonical.get("parent_events_used_for_training") != 0:
        raise IntegrityError("T10.3.8 canonical witness used parent events for training")
    if canonical.get("physical_actions_replayed") != 0:
        raise IntegrityError("T10.3.8 canonical witness replayed physical actions")
    if terminal.get("production_authority") is not False:
        raise IntegrityError("T10.3.8 source evidence has production authority")
    source_firewall = terminal.get("firewall", {})
    for field in (
        "ar25_opened",
        "automatic_retuning",
        "holdout_opened",
        "production_authority",
        "source_validation_opened",
    ):
        if source_firewall.get(field) is not False:
            raise IntegrityError(f"T10.3.8 source firewall drifted: {field}")

    collection_path = root / SOURCE_COLLECTION_MANIFEST
    if not collection_path.is_file():
        raise IntegrityError("lp85/su15 source collection manifest is absent")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    verify_signed(collection, "report_checksum")
    if collection.get("split") not in (None, "source_train"):
        raise IntegrityError("lp85/su15 collection has a non-source-train split")
    if collection.get("holdout_opened") or collection.get("ar25_opened"):
        raise IntegrityError("lp85/su15 source collection crossed a firewall")
    manifest_shards = {
        str(row.get("game_id")): str(row.get("sha256"))
        for row in collection.get("shards", ())
    }
    for short, expected in SOURCE_SHARD_SHA256.items():
        game_report = collection.get("game_reports", {}).get(short, {})
        if game_report.get("source_split") != "source_train":
            raise IntegrityError(f"source collection game split drifted: {short}")
        relative = SOURCE_SHARD_DIR / f"{short}.jsonl"
        path = root / relative
        if manifest_shards.get(short) != expected:
            raise IntegrityError(f"source collection shard digest drifted: {short}")
        if not path.is_file() or file_sha256(path) != expected:
            raise IntegrityError(f"source shard is absent or drifted: {short}")
    return observed


def source_success_journal_digest(root: Path) -> str:
    base = root / SOURCE_SUCCESS_JOURNAL_DIR
    rows = [
        {
            "relative_path": path.relative_to(base).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(base.rglob("*.json"))
        if path.is_file()
    ]
    if not rows:
        raise IntegrityError("T10.3.8 source-success journal is absent")
    digest = sha256_payload(rows)
    if digest != SOURCE_SUCCESS_JOURNAL_SHA256:
        raise IntegrityError("T10.3.8 source-success journal drifted")
    return digest


def source_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    paths = {
        "source_collection_manifest": SOURCE_COLLECTION_MANIFEST,
        "lp85_shard": SOURCE_SHARD_DIR / "lp85.jsonl",
        "su15_shard": SOURCE_SHARD_DIR / "su15.jsonl",
        **SOURCE_SUCCESS_ARTIFACT_PATHS,
    }
    for name, relative in paths.items():
        path = root / relative
        if not path.is_file():
            raise IntegrityError(f"required source artifact is absent: {name}")
        bindings[name] = {"path": relative.as_posix(), "sha256": file_sha256(path)}
    return bindings


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/causal_procedure_v10_3_12f.py",
        "theory/sage_t/t10_3_12f_protocol.py",
        "theory/sage_t/t10_3_12f_runtime.py",
        "tests/test_sage_t_causal_procedure_v10_3_12f.py",
        "tests/test_sage_t_t10_3_12f_protocol.py",
        "tests/test_sage_t_t10_3_12f_runtime.py",
        "reports/SAGE_T10_3_12F_CAUSAL_PROCEDURE_PROTOCOL.md",
        "reports/SAGE_T10_3_12F_CAUSAL_PROCEDURE_RUNBOOK.md",
    )
    output: dict[str, str] = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"T10.3.12f protocol dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


@dataclass(frozen=True)
class WorkSpec:
    phase: str
    game_id: str
    scope_index: int
    tie_break_seed: int
    arm: str
    reset_index: int
    action_budget: int

    @property
    def work_id(self) -> str:
        return sha256_payload(self.as_dict())

    @property
    def work_scope(self) -> str:
        return f"scope-{self.scope_index}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "game_id": self.game_id,
            "scope_index": self.scope_index,
            "work_scope": self.work_scope,
            "tie_break_seed": self.tie_break_seed,
            "arm": self.arm,
            "reset_index": self.reset_index,
            "action_budget": self.action_budget,
        }


def work_specs(phase: str) -> tuple[WorkSpec, ...]:
    if phase != "active-historical":
        raise ValueError(f"unsupported T10.3.12f physical phase: {phase}")
    rows: list[WorkSpec] = []
    for game_index, game in enumerate(TARGET_GAMES):
        for scope_index in WORK_SCOPES:
            rotation = (game_index + scope_index) % len(ARMS)
            ordered_arms = ARMS[rotation:] + ARMS[:rotation]
            for arm in ordered_arms:
                rows.append(
                    WorkSpec(
                        phase=phase,
                        game_id=game,
                        scope_index=scope_index,
                        tie_break_seed=312_600 + game_index * 16 + scope_index,
                        arm=arm,
                        reset_index=scope_index,
                        action_budget=ACTION_BUDGET,
                    )
                )
    return tuple(rows)


def maximum_actions_for_phase(phase: str) -> int:
    return sum(work.action_budget for work in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(work.action_budget) for work in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    if work.phase != "active-historical":
        raise ValueError("reset wall budget is active-historical only")
    return RESET_WALL_SECONDS


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent_state = verify_parent(root)
    source_state = verify_source_evidence(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "test_generalization_of_a_causal_identification_and_control_procedure",
        "hypotheses": {
            "procedure": (
                "a reset-local closed-loop causal identification verification control "
                "and revision procedure produces terminal progress across games"
            ),
            "source": (
                "a transfer-safe procedure prior compiled from lp85 and su15 improves "
                "that same causal procedure over uniform and permuted priors"
            ),
        },
        "post_hoc_historical_diagnostic": True,
        "claim_boundary": {
            "historical_candidate_identifiable": True,
            "prospective_generalization_proven": False,
            "independent_confirmation": False,
            "t10_3_13_holdout_authorized": False,
            "sequence_composition_authorized": False,
            "production_authority": False,
        },
        "code_hashes": _code_hashes(root),
        "parent_state": parent_state,
        "parent_artifacts": parent_artifact_bindings(root),
        "parent_journal_digest": parent_journal_digest(root),
        "source_state": source_state,
        "source_artifacts": source_artifact_bindings(root),
        "cli_phases": [
            "freeze",
            "status",
            "audit",
            "qa-source",
            "compile-prior",
            "evaluate-source",
            "preflight",
            "active-historical",
            "adjudicate",
            "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "artifact_contract": ARTIFACT_CONTRACT,
        "procedure_contract": {
            "phase_order": ["IDENTIFY", "VERIFY", "CONTROL", "REVISE", "ABSTAIN"],
            "model_families": list(MODEL_FAMILIES),
            "scoring_weights": {
                "information": 1.0,
                "repeatability": 0.75,
                "composability": 0.75,
                "risk": 1.0,
                "redundancy": 0.50,
            },
            "maximum_active_hypotheses": 8,
            "maximum_candidates_per_decision": 16,
            "control_posterior_threshold": 0.80,
            "control_margin_threshold": 0.20,
            "minimum_verification_contexts": 2,
            "mismatch_probability_threshold": 0.10,
            "stagnation_transition_limit": 4,
            "maximum_revisions": 2,
            "option_horizon": 16,
            "re_ground_after_every_transition": True,
            "fresh_posterior_per_work_spec": True,
            "legacy_fallback": False,
        },
        "arm_contract": {
            "shared_representation_learner_observations_thresholds_and_budgets": True,
            "source_closed_loop": {
                "prior": "lp85_su15_transfer_safe_procedure_prior",
                "revision": True,
            },
            "uniform_closed_loop": {
                "prior": "uniform_over_frozen_model_families",
                "revision": True,
            },
            "permuted_source_closed_loop": {
                "prior": "deterministic_family_permutation_of_source_prior",
                "same_norm_as_source": True,
                "same_entropy_as_source": True,
                "revision": True,
            },
            "source_open_loop": {
                "prior": "lp85_su15_transfer_safe_procedure_prior",
                "revision": False,
                "first_verified_hypothesis_locked": True,
            },
        },
        "source_policy": {
            "games": list(SOURCE_GAMES),
            "all_existing_signed_interventions_included": True,
            "successes_and_failures_included": True,
            "new_source_physical_actions": 0,
            "t10_3_8_winning_paths_terminal_link_only": True,
            "effects_reconstructed_from_frame_pairs": True,
            "legacy_effect_labels_trusted": False,
            "minimum_correspondence_confidence": 0.60,
            "persistent_one_to_one_correspondence_only": True,
            "free_space_and_background_excluded": True,
            "births_and_deaths_excluded_from_relation_deltas": True,
            "ambiguous_correspondences_rejected": True,
            "contradictory_relation_deltas_rejected": True,
            "validation_grouped_by_root_and_reset": True,
            "source_game_weights": {"lp85-305b61c3": 0.5, "su15-4c352900": 0.5},
            "transfer_payload_allows_only_procedure_family_weights": True,
            "forbidden_transfer_fields": [
                "game_id",
                "action_name",
                "action_arguments",
                "coordinates",
                "colors",
                "object_ids",
                "frame_hashes",
                "source_trajectories",
            ],
        },
        "matrix": {
            "games": list(TARGET_GAMES),
            "games_already_observed": True,
            "diagnostic_only": True,
            "arms": list(ARMS),
            "work_scopes": list(WORK_SCOPES),
            "work_scope_affects_only_tie_breaks": True,
            "work_scope_is_environment_seed": False,
            "arm_order": "game_and_scope_rotated_latin_square",
            "resets": TOTAL_RESETS,
            "maximum_actions_per_reset": ACTION_BUDGET,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "maximum_reset_wall_seconds": RESET_WALL_SECONDS,
            "maximum_global_wall_seconds": GLOBAL_WALL_SECONDS,
            "maximum_artifact_bytes": MAXIMUM_ARTIFACT_BYTES,
            "posterior_shared_between_work_specs": False,
            "stop_conditions": ["level_delta", "game_over", "abstention", "budget"],
        },
        "endpoints": {
            "primary": "utility_if_level_delta_else_zero",
            "utility_formula": "(action_budget + 1 - first_level_action) / action_budget",
            "terminal_level_delta_is_only_success_credit": True,
            "scope_aggregation": "mean_within_game_before_cross_game_test",
            "statistical_unit": "game",
            "secondary": [
                "prequential_effect_log_loss",
                "interventions_before_verification",
                "mismatch_rate",
                "revision_rate",
                "verified_context_diversity",
                "game_over_count",
                "noop_count",
                "abstention_count",
            ],
        },
        "gates": {
            "source_minimum_effect_modes_per_game": 2,
            "source_maximum_single_label_fraction_exclusive": 0.95,
            "source_prior_must_be_non_uniform": True,
            "source_prior_maximum_family_weight": 0.70,
            "source_contribution_per_game": 0.50,
            "minimum_log_loss_improvement_over_permuted_each_source": 0.05,
            "minimum_identification_intervention_reduction_on_one_source": 0.20,
            "maximum_identification_regression_on_other_source": 0.0,
            "all_receipts_required": TOTAL_RESETS,
            "distinct_target_games_required": len(TARGET_GAMES),
            "minimum_candidate_success_games": 2,
            "minimum_identification_verified_games": 2,
            "minimum_identification_better_games_each_control": 5,
            "exact_sign_permutation_unit": "game_mean_over_four_scopes",
            "holm_familywise_alpha": 0.05,
            "source_candidate_must_beat": [
                "uniform_closed_loop",
                "permuted_source_closed_loop",
                "source_open_loop",
            ],
            "generic_candidate_must_beat": ["source_open_loop"],
            "maximum_illegal_actions": 0,
            "maximum_legacy_fallback_actions": 0,
            "maximum_physical_replay_actions": 0,
        },
        "adjudication_contract": {
            "source_informed_candidate": {
                "arm": "source_closed_loop",
                "minimum_success_games": 2,
                "must_beat_utility_arms": [
                    "uniform_closed_loop",
                    "permuted_source_closed_loop",
                    "source_open_loop",
                ],
                "test": "exact_paired_sign_permutation_by_game",
                "multiplicity": "holm_0.05",
                "pass_verdict": (
                    "PASS_T10_3_12F_HISTORICAL_SOURCE_INFORMED_"
                    "CAUSAL_PROCEDURE_CANDIDATE"
                ),
            },
            "generic_candidate": {
                "arm": "uniform_closed_loop",
                "minimum_success_games": 2,
                "must_beat_utility_arms": ["source_open_loop"],
                "source_over_uniform_must_not_be_significant": True,
                "test": "exact_paired_sign_permutation_by_game",
                "multiplicity": "holm_0.05",
                "pass_verdict": (
                    "PASS_T10_3_12F_HISTORICAL_GENERIC_"
                    "CAUSAL_PROCEDURE_CANDIDATE"
                ),
            },
            "success_counts_only_level_delta": True,
            "no_pass_promotes_a_program": True,
        },
        "preflight_contract": {
            "d4_invariance": True,
            "palette_invariance": True,
            "candidate_order_invariance": True,
            "object_identifier_invariance": True,
            "birth_death_exclusion": True,
            "ambiguous_correspondence_rejection": True,
            "universal_label_stop": True,
            "posterior_updates_after_sealed_event": True,
            "mismatch_revision_and_abstention": True,
            "permuted_prior_ranking_inversion": True,
            "serialization_transfer_safe": True,
            "durable_resume_without_replay": True,
        },
        "firewall": {
            "new_games_opened": False,
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "legacy_fallback_authorized": False,
            "t10_3_13_authorized": False,
            "source_games_physical_actions_authorized": False,
            "source_shards_offline_projection_only": True,
            "source_shards_physical_replay_authorized": False,
            "t10_3_12e_events_training_authorized": False,
            "t10_3_12c_to_e_events_initialization_authorized": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "post_action_observe_required": True,
            "physical_replay": False,
            "write_once": True,
            "fresh_controller_per_work_spec": True,
            "bounded_resume_from_sealed_receipts_only": True,
            "planned_abstention_is_complete_result": True,
        },
        "negative_result_policy": {
            "no_post_freeze_repair": True,
            "no_automatic_retuning": True,
            "source_qa_miss_verdict": "CAUSAL_LABEL_QA_MISS",
            "source_identification_miss_verdict": (
                "PROCEDURE_NOT_SOURCE_IDENTIFIABLE"
            ),
            "single_game_success_verdict": "SINGLE_GAME_EFFECT_ONLY",
            "identification_without_control_verdict": (
                "CAUSAL_IDENTIFICATION_WITHOUT_CONTROL"
            ),
            "source_prior_nonspecific_verdict": "SOURCE_PRIOR_NOT_SPECIFIC",
            "no_progress_verdict": "CAUSAL_PROCEDURE_NO_TARGET_PROGRESS",
            "pass_is_historical_candidate_only": True,
            "pass_authorizes_only_t10_3_13_preregistration": True,
            "no_program_promotion": True,
        },
        "output_directory": DEFAULT_OUTPUT_DIR.as_posix(),
    }
    return _signed(core, "manifest_checksum")


def freeze_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
    receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = build_manifest(root)
    write_json_once(manifest_path or root / DEFAULT_MANIFEST_PATH, manifest)
    receipt = _signed(
        {
            "format_version": "sage-t10.3.12f-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_checksum": manifest["parent_state"][
                "terminal_report_checksum"
            ],
            "parent_journal_digest": manifest["parent_journal_digest"],
            "source_state": manifest["source_state"],
            "historical_diagnostic_only": True,
            "maximum_resets": TOTAL_RESETS,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "physical_actions_at_freeze": 0,
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "t10_3_13_authorized": False,
        },
        "receipt_checksum",
    )
    write_json_once(receipt_path or root / DEFAULT_FREEZE_RECEIPT_PATH, receipt)
    return manifest, receipt


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.12f manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.12f manifest format drifted")
    if verify_parent(root) != payload.get("parent_state"):
        raise IntegrityError("T10.3.12e parent state changed after T10.3.12f freeze")
    if verify_source_evidence(root) != payload.get("source_state"):
        raise IntegrityError("lp85/su15 source evidence changed after T10.3.12f freeze")
    for group in ("parent_artifacts", "source_artifacts"):
        for name, binding in payload.get(group, {}).items():
            bound_path = root / str(binding["path"])
            if not bound_path.is_file() or file_sha256(bound_path) != binding["sha256"]:
                raise IntegrityError(f"frozen T10.3.12f binding drifted: {group}.{name}")
    if payload.get("parent_journal_digest") != parent_journal_digest(root):
        raise IntegrityError("T10.3.12e journal changed after T10.3.12f freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.12f code changed after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.12f freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.12f freeze receipt is detached")
    if receipt.get("holdout_opened") or receipt.get("t10_3_13_authorized"):
        raise IntegrityError("T10.3.12f receipt opened prospective authority")
    return payload


__all__ = [
    "ACTION_BUDGET",
    "ARMS",
    "ARTIFACT_CONTRACT",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_PARENT",
    "EXPECTED_SOURCE_SUCCESS",
    "FORMAT_VERSION",
    "GLOBAL_WALL_SECONDS",
    "IntegrityError",
    "MANIFEST_STATUS",
    "MAXIMUM_ARTIFACT_BYTES",
    "MODEL_FAMILIES",
    "PARENT_ARTIFACT_PATHS",
    "PARENT_OUTPUT_DIR",
    "RESET_WALL_SECONDS",
    "SOURCE_COLLECTION_MANIFEST",
    "SOURCE_GAMES",
    "SOURCE_SHARD_DIR",
    "SOURCE_SHARD_SHA256",
    "SOURCE_SUCCESS_ARTIFACT_PATHS",
    "SOURCE_SUCCESS_JOURNAL_DIR",
    "SOURCE_SUCCESS_JOURNAL_SHA256",
    "SOURCE_SUCCESS_OUTPUT_DIR",
    "ScientificGateMiss",
    "TARGET_GAMES",
    "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS",
    "WORK_SCOPES",
    "WorkSpec",
    "build_manifest",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "maximum_actions_for_phase",
    "maximum_actions_for_specs",
    "parent_artifact_bindings",
    "parent_journal_digest",
    "reset_wall_seconds",
    "sha256_payload",
    "source_artifact_bindings",
    "source_success_journal_digest",
    "verify_parent",
    "verify_signed",
    "verify_source_evidence",
    "work_specs",
    "write_json_once",
]
