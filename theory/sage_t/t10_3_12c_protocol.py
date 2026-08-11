"""Preregistered cross-game falsification protocol for SAGE.T10.3.12c."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_12b_protocol as parent
from .cross_game_transfer_v10_3_12c import ARMS, FACTORS, sha256_payload

FORMAT_VERSION = "sage-t10.3.12c-cross-game-falsification-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_12C_TARGET_RESET"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_12c_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name("sage_t10_3_12c_freeze_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_12c_cross_game_factor_falsification"

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
TARGET_SHORT_IDS = tuple(game.split("-", 1)[0] for game in TARGET_GAMES)
TARGET_SHARD_SHA256 = {
    "bp35": "2c2d97fe934543f6fc19250c8f73f314120074c0dfbbe442e1704db7bed6c04b",
    "cd82": "eee5871f51cfc58d9e0441784083453629cdb65df2824f2f2013582ba23410ea",
    "dc22": "7e94dfaf7609c45f9d81e8016ed8d246668094e751cc17112651687213af23d6",
    "g50t": "314cceaae603ecc82b339ed99934cc8d6bce4886768939132cdcebbb6fd8acff",
    "ka59": "879fa2c9aa67b317b88ba15e95bd0066b4a9f3b5feff0eede19de126275d0f5d",
    "lf52": "0e1b171ac09f23734f8f8cb8686c176cc0f6ff02b2dfbd4891e8079f9fd24aa9",
    "sp80": "aab95e73dae8957b70ba32380ac17b9e45e4c70c7786628d3c1b4d73fcdc6db4",
    "tr87": "53229b7b0d3584960a86cafbc2e89426e54eaa4c559a5762b383d9da26f5cdfa",
    "tu93": "0b0d9fede128c1cddb93a0ec95725b5cc29f6fda7512f0e0c5613170185eae49",
}
SOURCE_COLLECTION_MANIFEST = Path(
    "training/sage12/bound_mechanic_pilot_v4_3/source_train_collection_manifest.json"
)
SOURCE_SHARD_DIR = Path("training/sage12/bound_mechanic_pilot_v4_3/source_train_shards")
PARENT_OUTPUT_DIR = Path("training/sage_t/t10_3_12b_factorial_invariant_identification")
PARENT_ARTIFACT_PATHS = {
    "manifest": Path("theory/sage_t/sage_t10_3_12b_protocol_manifest.json"),
    "freeze_receipt": Path("theory/sage_t/sage_t10_3_12b_freeze_receipt.json"),
    "factor_registry": PARENT_OUTPUT_DIR / "factor_registry.json",
    "intervention_report": PARENT_OUTPUT_DIR / "factorial_intervention_report.json",
    "adjudication": PARENT_OUTPUT_DIR / "factor_adjudication_report.json",
    "terminal": PARENT_OUTPUT_DIR / "terminal_report.json",
}
EXPECTED_PARENT = {
    "manifest_checksum": "cec956171b7abcf752fe2809d6545b951f089eba1dee7aa4406bc6fbba939699",
    "terminal_report_checksum": "b01e2d764ebd4cfc3a3f9528cab9333198dc032e07a7e51ae3f1e180913b4ce2",
    "verdict": "PASS_T10_3_12B_TRANSFERABLE_FACTOR_CANDIDATES_IDENTIFIED",
    "identified_factor_candidates": list(FACTORS),
}

ACTION_BUDGET = 16
RESET_WALL_SECONDS = 180.0
TOTAL_RESETS = len(TARGET_GAMES) * len(ARMS)
TOTAL_MAXIMUM_ACTIONS = TOTAL_RESETS * ACTION_BUDGET

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
write_json_once = parent.write_json_once

ARTIFACT_CONTRACT = {
    "audit-parent": {
        "path": "parent_factor_candidate_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
    },
    "preflight": {
        "path": "cross_game_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
    },
    "audit-targets": {
        "path": "target_schema_inventory.json",
        "checksum_field": "inventory_checksum",
        "gate_field": "passed",
    },
    "compile-transfer": {
        "path": "cross_game_factor_registry.json",
        "checksum_field": "registry_checksum",
        "gate_field": None,
    },
    "active-transfer": {
        "path": "active_cross_game_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "collection_complete",
    },
    "adjudicate": {
        "path": "cross_game_adjudication_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
    },
    "report": {
        "path": "terminal_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
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


def _parent_payload(root: Path, name: str, checksum_field: str) -> dict[str, Any]:
    path = root / PARENT_ARTIFACT_PATHS[name]
    if not path.is_file():
        raise IntegrityError(f"required T10.3.12b artifact is absent: {name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def verify_parent(root: Path) -> dict[str, Any]:
    parent_manifest = parent.load_manifest(root)
    terminal = _parent_payload(root, "terminal", "report_checksum")
    adjudication = _parent_payload(root, "adjudication", "report_checksum")
    registry = _parent_payload(root, "factor_registry", "registry_checksum")
    _parent_payload(root, "intervention_report", "report_checksum")
    observed = {
        "manifest_checksum": parent_manifest.get("manifest_checksum"),
        "terminal_report_checksum": terminal.get("report_checksum"),
        "verdict": terminal.get("verdict"),
        "identified_factor_candidates": adjudication.get("identified_factor_candidates"),
    }
    if observed != EXPECTED_PARENT:
        raise IntegrityError("T10.3.12b parent result diverged from the frozen PASS")
    if adjudication.get("cross_game_preregistration_authorized") is not True:
        raise IntegrityError("T10.3.12b did not authorize cross-game preregistration")
    if adjudication.get("cross_game_generalization_proven") is not False:
        raise IntegrityError("T10.3.12b claim boundary drifted")
    if int(registry.get("local_support_total", -1)) != 0:
        raise IntegrityError("T10.3.12b factor registry contains active support")
    return observed


def parent_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: {"path": relative.as_posix(), "sha256": file_sha256(root / relative)}
        for name, relative in PARENT_ARTIFACT_PATHS.items()
    }


def target_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    collection = root / SOURCE_COLLECTION_MANIFEST
    if not collection.is_file():
        raise IntegrityError("source-train collection manifest is absent")
    collection_payload = json.loads(collection.read_text(encoding="utf-8"))
    if collection_payload.get("split") != "source_train":
        raise IntegrityError("target inventory is not source_train")
    if collection_payload.get("holdout_opened") or collection_payload.get("ar25_opened"):
        raise IntegrityError("source-train inventory crossed a firewall")
    manifest_shards = {
        str(row.get("game_id")): str(row.get("sha256"))
        for row in collection_payload.get("shards", ())
    }
    bindings = {
        "source_collection_manifest": {
            "path": SOURCE_COLLECTION_MANIFEST.as_posix(),
            "sha256": file_sha256(collection),
        }
    }
    for game in TARGET_GAMES:
        short, version = game.split("-", 1)
        if manifest_shards.get(short) != TARGET_SHARD_SHA256[short]:
            raise IntegrityError(f"source-train shard manifest mismatch: {short}")
        shard = SOURCE_SHARD_DIR / f"{short}.jsonl"
        python = Path("environment_files") / short / version / f"{short}.py"
        metadata = Path("environment_files") / short / version / "metadata.json"
        for label, relative in (("shard", shard), ("environment", python), ("metadata", metadata)):
            path = root / relative
            if not path.is_file():
                raise IntegrityError(f"target artifact is absent: {relative.as_posix()}")
            digest = file_sha256(path)
            if label == "shard" and digest != TARGET_SHARD_SHA256[short]:
                raise IntegrityError(f"source-train shard drifted: {short}")
            bindings[f"{short}_{label}"] = {
                "path": relative.as_posix(),
                "sha256": digest,
            }
    return bindings


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/cross_game_transfer_v10_3_12c.py",
        "theory/sage_t/t10_3_12c_protocol.py",
        "theory/sage_t/t10_3_12c_runtime.py",
        "tests/test_sage_t_cross_game_transfer_v10_3_12c.py",
        "tests/test_sage_t_t10_3_12c_protocol.py",
        "tests/test_sage_t_t10_3_12c_runtime.py",
        "reports/SAGE_T10_3_12C_CROSS_GAME_PROTOCOL.md",
        "reports/SAGE_T10_3_12C_CROSS_GAME_RUNBOOK.md",
    )
    output = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"T10.3.12c protocol dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


@dataclass(frozen=True)
class WorkSpec:
    phase: str
    game_id: str
    seed: int
    arm: str
    reset_index: int
    action_budget: int

    @property
    def work_id(self) -> str:
        return sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "game_id": self.game_id,
            "seed": self.seed,
            "arm": self.arm,
            "reset_index": self.reset_index,
            "action_budget": self.action_budget,
        }


def work_specs(phase: str) -> tuple[WorkSpec, ...]:
    if phase != "active-transfer":
        raise ValueError(f"unsupported T10.3.12c physical phase: {phase}")
    rows = []
    for game_index, game in enumerate(TARGET_GAMES):
        rotation = game_index % len(ARMS)
        ordered_arms = ARMS[rotation:] + ARMS[:rotation]
        for arm in ordered_arms:
            rows.append(
                WorkSpec(
                    phase=phase,
                    game_id=game,
                    seed=3621 + game_index,
                    arm=arm,
                    reset_index=0,
                    action_budget=ACTION_BUDGET,
                )
            )
    return tuple(rows)


def maximum_actions_for_phase(phase: str) -> int:
    return sum(work.action_budget for work in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(work.action_budget) for work in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    if work.phase != "active-transfer":
        raise ValueError("reset wall budget is active-transfer only")
    return RESET_WALL_SECONDS


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent_state = verify_parent(root)
    parent_bindings = parent_artifact_bindings(root)
    target_bindings = target_artifact_bindings(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "falsify_t10_3_12b_factor_candidates_on_all_remaining_source_train_games",
        "hypothesis": (
            "the frozen operator role transition and termination factors yield "
            "cross-game progress that cannot be explained by source-free bounded search"
        ),
        "claim_boundary": {
            "cross_game_factor_evidence_possible": True,
            "sequence_composition_authorized": False,
            "source_validation_authorized": False,
            "production_authority": False,
        },
        "code_hashes": _code_hashes(root),
        "parent_state": parent_state,
        "parent_artifacts": parent_bindings,
        "target_artifacts": target_bindings,
        "cli_phases": [
            "freeze", "status", "audit-parent", "preflight", "audit-targets",
            "compile-transfer", "active-transfer", "adjudicate", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "artifact_contract": ARTIFACT_CONTRACT,
        "matrix": {
            "games": list(TARGET_GAMES),
            "selection_rule": "all_remaining_source_train_games_excluding_lp85_su15",
            "arms": list(ARMS),
            "ablated_factors": list(FACTORS),
            "one_reset_per_game_arm": True,
            "labels_seed_environment": False,
            "arm_order": "game_rotated_latin_order",
            "resets": TOTAL_RESETS,
            "maximum_actions_per_reset": ACTION_BUDGET,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "maximum_legal_candidates_processed_per_decision": 512,
            "maximum_reset_wall_seconds": RESET_WALL_SECONDS,
            "maximum_global_wall_seconds": 10_800,
            "maximum_artifact_bytes": 20 * 1024 * 1024,
        },
        "gates": {
            "all_target_games_included": 9,
            "all_receipts_required": TOTAL_RESETS,
            "minimum_factorized_applicable_games": 3,
            "minimum_factorized_success_games": 3,
            "minimum_factorized_success_rate_on_applicable": 2 / 3,
            "minimum_paired_ablation_advantage": 2,
            "maximum_reverse_paired_ablation_wins": 0,
            "generic_equal_success_action_ratio_maximum": 0.75,
            "minimum_supported_factors": 1,
        },
        "firewall": {
            "lp85_su15_target_scoring_opened": False,
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "legacy_fallback_authorized": False,
            "t10_3_12b_grounded_evidence_authorized": False,
            "target_shards_used_for_action_schema_audit_only": True,
            "target_shard_effects_or_outcomes_read": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "write_once": True,
            "fresh_controller_per_game_arm": True,
            "planned_abstention_is_complete_zero_action_result": True,
        },
        "negative_result_policy": {
            "no_post_freeze_repair": True,
            "operator_coverage_miss_is_scientific_result": True,
            "generic_tie_is_negative": True,
            "partial_factor_support_reported_by_name": True,
            "no_program_promotion": True,
            "pass_authorizes_only_independent_cross_game_reproduction": True,
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
            "format_version": "sage-t10.3.12c-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_checksum": manifest["parent_state"]["terminal_report_checksum"],
            "target_count": len(TARGET_GAMES),
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "physical_actions_at_freeze": 0,
            "sequence_games_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
        "receipt_checksum",
    )
    write_json_once(receipt_path or root / DEFAULT_FREEZE_RECEIPT_PATH, receipt)
    return manifest, receipt


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.12c manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.12c manifest format drifted")
    if verify_parent(root) != payload.get("parent_state"):
        raise IntegrityError("T10.3.12b parent state changed after freeze")
    for group in ("parent_artifacts", "target_artifacts"):
        for name, binding in payload.get(group, {}).items():
            path = root / str(binding["path"])
            if not path.is_file() or file_sha256(path) != binding["sha256"]:
                raise IntegrityError(f"frozen T10.3.12c binding drifted: {name}")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.12c code changed after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.12c freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.12c freeze receipt is detached")
    return payload


__all__ = [
    "ACTION_BUDGET", "ARMS", "ARTIFACT_CONTRACT", "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH", "DEFAULT_OUTPUT_DIR", "EXPECTED_PARENT", "FACTORS",
    "FORMAT_VERSION", "IntegrityError", "MANIFEST_STATUS", "PARENT_ARTIFACT_PATHS",
    "PARENT_OUTPUT_DIR", "RESET_WALL_SECONDS", "SOURCE_COLLECTION_MANIFEST",
    "SOURCE_SHARD_DIR", "ScientificGateMiss", "TARGET_GAMES", "TARGET_SHARD_SHA256",
    "TARGET_SHORT_IDS", "TOTAL_MAXIMUM_ACTIONS", "TOTAL_RESETS", "WorkSpec",
    "build_manifest", "file_sha256", "freeze_manifest", "load_manifest",
    "maximum_actions_for_phase", "maximum_actions_for_specs", "reset_wall_seconds",
    "sha256_payload", "target_artifact_bindings", "verify_parent", "verify_signed",
    "work_specs", "write_json_once",
]
