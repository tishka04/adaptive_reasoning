"""Locked prospective confirmation protocol for SAGE T10.3.13.

Importing this module never reads a protected frame.  A manifest can be built
only after a positive T10.3.12f terminal report and a separate, explicit
holdout-authorization receipt have both been verified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_12f_protocol as parent

FORMAT_VERSION = "sage-t10.3.13-prospective-causal-procedure-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_13_PROTECTED_RESET"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_13_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name(
    "sage_t10_3_13_freeze_receipt.json"
)
DEFAULT_AUTHORIZATION_PATH = Path(__file__).with_name(
    "sage_t10_3_13_holdout_authorization.json"
)
DEFAULT_OUTPUT_DIR = Path("training/sage_t/t10_3_13_prospective_causal_procedure")
PARENT_OUTPUT_DIR = parent.DEFAULT_OUTPUT_DIR
PARENT_TERMINAL_PATH = PARENT_OUTPUT_DIR / "terminal_report.json"
PARENT_PRIOR_PATH = PARENT_OUTPUT_DIR / "causal_procedure_prior.json"

AUTHORIZATION_PHRASE = "I AUTHORIZE T10.3.13 PROTECTED HOLDOUT"
PROTECTED_GAMES = (
    "s5i5",
    "vc33",
    "m0r0",
    "sk48",
    "r11l",
)
SOURCE_PASS = (
    "PASS_T10_3_12F_HISTORICAL_SOURCE_INFORMED_"
    "CAUSAL_PROCEDURE_CANDIDATE"
)
GENERIC_PASS = (
    "PASS_T10_3_12F_HISTORICAL_GENERIC_CAUSAL_PROCEDURE_CANDIDATE"
)
ACTION_BUDGET = 48
RESET_WALL_SECONDS = 180.0
GLOBAL_WALL_SECONDS = 2 * 60 * 60
MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
TOTAL_RESETS = len(PROTECTED_GAMES) * 2
TOTAL_MAXIMUM_ACTIONS = TOTAL_RESETS * ACTION_BUDGET

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
sha256_payload = parent.sha256_payload
write_json_once = parent.write_json_once

ARTIFACT_CONTRACT = {
    "audit-parent": {
        "path": "parent_candidate_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "preflight": {
        "path": "prospective_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
        "role": "report",
    },
    "active-confirmation": {
        "path": "active_confirmation_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "collection_complete",
        "role": "report",
    },
    "adjudicate": {
        "path": "prospective_adjudication.json",
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


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise IntegrityError(f"invalid {checksum_field}")


def candidate_pair(verdict: str) -> tuple[str, str]:
    if verdict == SOURCE_PASS:
        return "source_closed_loop", "uniform_closed_loop"
    if verdict == GENERIC_PASS:
        return "uniform_closed_loop", "source_open_loop"
    raise ScientificGateMiss("T10.3.12f did not nominate a prospective candidate")


def verify_parent_candidate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = parent.load_manifest(root)
    terminal_path = root / PARENT_TERMINAL_PATH
    prior_path = root / PARENT_PRIOR_PATH
    if not terminal_path.is_file() or not prior_path.is_file():
        raise IntegrityError("T10.3.12f terminal report or procedure prior is absent")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    verify_signed(terminal, "report_checksum")
    verify_signed(prior, "prior_checksum")
    if terminal.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise IntegrityError("T10.3.12f terminal report is detached from its manifest")
    if terminal.get("passed") is not True:
        raise ScientificGateMiss("T10.3.12f did not pass")
    candidate, control = candidate_pair(str(terminal.get("verdict", "")))
    artifacts = terminal.get("artifacts", {})
    if artifacts.get("compile-prior") != prior.get("prior_checksum"):
        raise IntegrityError("T10.3.12f terminal report is detached from its prior")
    for field in (
        "holdout_opened",
        "ar25_opened",
        "production_authority",
        "sequence_games_opened",
    ):
        if terminal.get(field) is not False:
            raise IntegrityError(f"T10.3.12f firewall drifted: {field}")
    return {
        "manifest_checksum": manifest["manifest_checksum"],
        "terminal_report_checksum": terminal["report_checksum"],
        "prior_checksum": prior["prior_checksum"],
        "verdict": terminal["verdict"],
        "candidate_arm": candidate,
        "control_arm": control,
    }


def authorize_holdout(
    root: Path,
    *,
    acknowledgement: str,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Write the zero-action authorization receipt after explicit acknowledgement."""

    if acknowledgement != AUTHORIZATION_PHRASE:
        raise IntegrityError("protected holdout acknowledgement does not match")
    state = verify_parent_candidate(root)
    receipt = _signed(
        {
            "format_version": "sage-t10.3.13-holdout-authorization-v1",
            "explicit_authorization": True,
            "acknowledgement_sha256": sha256_payload(AUTHORIZATION_PHRASE),
            "parent_state": state,
            "protected_games": list(PROTECTED_GAMES),
            "protected_frames_read_at_authorization": 0,
            "physical_actions_at_authorization": 0,
            "one_final_confirmation_only": True,
            "ar25_opened": False,
            "production_authority": False,
        },
        "authorization_checksum",
    )
    write_json_once(authorization_path or root / DEFAULT_AUTHORIZATION_PATH, receipt)
    return receipt


def load_authorization(root: Path) -> dict[str, Any]:
    path = root.resolve() / DEFAULT_AUTHORIZATION_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.13 protected holdout is not explicitly authorized")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "authorization_checksum")
    if payload.get("explicit_authorization") is not True:
        raise IntegrityError("T10.3.13 authorization polarity is invalid")
    if payload.get("parent_state") != verify_parent_candidate(root):
        raise IntegrityError("T10.3.13 authorization is detached from T10.3.12f")
    if payload.get("protected_frames_read_at_authorization") != 0:
        raise IntegrityError("protected frames were read before T10.3.13 freeze")
    return payload


@dataclass(frozen=True)
class WorkSpec:
    phase: str
    game_id: str
    arm: str
    role: str
    reset_index: int
    action_budget: int

    @property
    def work_id(self) -> str:
        return sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "game_id": self.game_id,
            "arm": self.arm,
            "role": self.role,
            "reset_index": self.reset_index,
            "action_budget": self.action_budget,
        }


def work_specs(phase: str, *, candidate: str, control: str) -> tuple[WorkSpec, ...]:
    if phase != "active-confirmation":
        raise ValueError(f"unsupported T10.3.13 physical phase: {phase}")
    rows: list[WorkSpec] = []
    for index, game in enumerate(PROTECTED_GAMES):
        ordered = ((candidate, "candidate"), (control, "control"))
        if index % 2:
            ordered = tuple(reversed(ordered))
        for arm, role in ordered:
            rows.append(
                WorkSpec(
                    phase=phase,
                    game_id=game,
                    arm=arm,
                    role=role,
                    reset_index=0,
                    action_budget=ACTION_BUDGET,
                )
            )
    return tuple(rows)


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/causal_procedure_v10_3_12f.py",
        "theory/sage_t/t10_3_12f_protocol.py",
        "theory/sage_t/t10_3_12f_runtime.py",
        "theory/sage_t/t10_3_13_protocol.py",
        "theory/sage_t/t10_3_13_runtime.py",
        "tests/test_sage_t_t10_3_13_protocol.py",
        "tests/test_sage_t_t10_3_13_runtime.py",
        "reports/SAGE_T10_3_13_PROSPECTIVE_CAUSAL_PROCEDURE_PROTOCOL.md",
    )
    output: dict[str, str] = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"T10.3.13 dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    state = verify_parent_candidate(root)
    authorization = load_authorization(root)
    specs = work_specs(
        "active-confirmation",
        candidate=state["candidate_arm"],
        control=state["control_arm"],
    )
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "prospective_confirmation_of_frozen_causal_procedure",
        "parent_state": state,
        "authorization_checksum": authorization["authorization_checksum"],
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "status",
            "authorize-holdout",
            "freeze",
            "audit-parent",
            "preflight",
            "active-confirmation",
            "adjudicate",
            "report",
        ],
        "artifact_contract": ARTIFACT_CONTRACT,
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": {
            "games": list(PROTECTED_GAMES),
            "candidate_arm": state["candidate_arm"],
            "control_arm": state["control_arm"],
            "resets": len(specs),
            "maximum_actions_per_reset": ACTION_BUDGET,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "maximum_reset_wall_seconds": RESET_WALL_SECONDS,
            "maximum_global_wall_seconds": GLOBAL_WALL_SECONDS,
            "maximum_artifact_bytes": MAXIMUM_ARTIFACT_BYTES,
            "one_final_confirmation_only": True,
            "initial_hash_must_match_within_game_pair": True,
        },
        "gates": {
            "minimum_candidate_success_games": 3,
            "minimum_net_success_advantage": 2,
            "minimum_games_with_higher_utility": 4,
            "maximum_games_with_lower_utility": 0,
            "minimum_games_with_better_log_loss": 4,
            "maximum_illegal_actions": 0,
            "maximum_legacy_fallback_actions": 0,
            "maximum_physical_replay_actions": 0,
        },
        "firewall": {
            "holdout_authorized": True,
            "holdout_opened_at_freeze": False,
            "ar25_opened": False,
            "source_validation_opened": False,
            "sequence_games_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "program_promotion_authorized": False,
        },
        "output_directory": DEFAULT_OUTPUT_DIR.as_posix(),
    }
    return _signed(core, "manifest_checksum")


def freeze_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = build_manifest(root)
    write_json_once(root / DEFAULT_MANIFEST_PATH, manifest)
    receipt = _signed(
        {
            "format_version": "sage-t10.3.13-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "authorization_checksum": manifest["authorization_checksum"],
            "parent_state": manifest["parent_state"],
            "physical_actions_at_freeze": 0,
            "protected_frames_read_at_freeze": 0,
            "holdout_opened_at_freeze": False,
            "ar25_opened": False,
            "production_authority": False,
        },
        "receipt_checksum",
    )
    write_json_once(root / DEFAULT_FREEZE_RECEIPT_PATH, receipt)
    return manifest, receipt


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.13 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.13 manifest format drifted")
    if payload.get("parent_state") != verify_parent_candidate(root):
        raise IntegrityError("T10.3.12f candidate changed after T10.3.13 freeze")
    authorization = load_authorization(root)
    if payload.get("authorization_checksum") != authorization.get(
        "authorization_checksum"
    ):
        raise IntegrityError("T10.3.13 manifest is detached from authorization")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.13 code changed after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.13 freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.13 freeze receipt is detached")
    if receipt.get("protected_frames_read_at_freeze") != 0:
        raise IntegrityError("protected frames were read before freeze")
    return payload


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(work.action_budget) for work in specs)


__all__ = [
    "ACTION_BUDGET",
    "ARTIFACT_CONTRACT",
    "AUTHORIZATION_PHRASE",
    "DEFAULT_AUTHORIZATION_PATH",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "GENERIC_PASS",
    "IntegrityError",
    "PROTECTED_GAMES",
    "SOURCE_PASS",
    "ScientificGateMiss",
    "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS",
    "WorkSpec",
    "authorize_holdout",
    "build_manifest",
    "candidate_pair",
    "freeze_manifest",
    "load_authorization",
    "load_manifest",
    "maximum_actions_for_specs",
    "verify_parent_candidate",
    "verify_signed",
    "work_specs",
]
