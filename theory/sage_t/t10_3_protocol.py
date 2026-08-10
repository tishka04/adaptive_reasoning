"""Frozen source-only protocol for the SAGE.T10.3 goal-progress pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel_protocol

FORMAT_VERSION = "sage-t10.3-goal-progress-protocol-v1"
HANDOFF_FORMAT_VERSION = "sage-t10.3-goal-progress-handoff-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_SOURCE_ACTION"

T10_2_9_TERMINAL_CHECKSUM = (
    "a94d09c5264c941b23d1e68e13e9615e314f35abb1490e989ce6e44da175d3f5"
)
T10_2_9_QA_CHECKSUM = (
    "eff6b2d6aa27c9e5830c540783f82c967c51a14b86371c771fa4e16bc96e8a5e"
)
T10_0B_REPORT_CHECKSUM = (
    "a72bff60c6fec8a7fc1d3a7b4ecebabb5d84c39bc80bfc57ed641e0080075fbc"
)

SOURCE_GAMES = ("bp35-0a0ad940", "lp85-305b61c3", "su15-4c352900")
POSITIVE_WITNESS_GAMES = ("lp85-305b61c3", "su15-4c352900")
PANEL_SEEDS = (3101, 3102, 3103, 3104)
CONFIRMATION_SEEDS = (3111, 3112)
PANEL_ARMS = (
    "canonical_option",
    "binding_swap",
    "option_intervention",
    "capacity_matched_independent",
)
CONFIRMATION_CONTROLLERS = ("learned", "capacity_matched_independent")
MAXIMUM_ACTIONS_PER_RESET = 16
PANEL_RESETS = len(SOURCE_GAMES) * len(PANEL_SEEDS) * len(PANEL_ARMS)
CONFIRMATION_RESETS = (
    len(SOURCE_GAMES) * len(CONFIRMATION_SEEDS) * len(CONFIRMATION_CONTROLLERS)
)
PANEL_MAXIMUM_ACTIONS = PANEL_RESETS * MAXIMUM_ACTIONS_PER_RESET
CONFIRMATION_MAXIMUM_ACTIONS = CONFIRMATION_RESETS * MAXIMUM_ACTIONS_PER_RESET
TOTAL_RESETS = PANEL_RESETS + CONFIRMATION_RESETS
TOTAL_MAXIMUM_ACTIONS = PANEL_MAXIMUM_ACTIONS + CONFIRMATION_MAXIMUM_ACTIONS

PHASES = (
    "freeze",
    "status",
    "audit",
    "collect",
    "compile",
    "fit",
    "confirm",
    "report",
)
EXCLUSIVE_VERDICTS = (
    "PROVENANCE_INVALID",
    "ROOTING_MISS",
    "WITNESS_REPRODUCTION_MISS",
    "QA_MISS",
    "CAUSAL_SEMANTICS_MISS",
    "OPTION_INDUCTION_MISS",
    "SOURCE_CONFIRMATION_MISS",
    "PASS_T10_3_SOURCE_PILOT",
)

DEFAULT_MANIFEST_RELATIVE_PATH = Path("theory/sage_t/sage_t10_3_protocol_manifest.json")
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_HANDOFF_RELATIVE_PATH = Path("theory/sage_t/sage_t10_3_handoff_receipt.json")
DEFAULT_HANDOFF_PATH = Path(__file__).with_name(DEFAULT_HANDOFF_RELATIVE_PATH.name)
DEFAULT_OUTPUT_ROOT = Path("training") / "sage_t" / "t10_3_goal_progress_pilot"
T10_2_9_ROOT = Path("training") / "sage_t" / "t10_2_9_offline_qa"
T10_2_7_LEDGER = (
    Path("training")
    / "sage_t"
    / "t10_2_7_event_seal_recovery"
    / "accepted_source_events.jsonl"
)
T10_0B_REPORT = Path("training") / "sage_t" / "progress_witness_v10_0b" / "report.json"

DEFAULT_CODE_FILES = (
    "theory/sage_t/frame_adapters_v10_3.py",
    "theory/sage_t/t10_3_protocol.py",
    "theory/sage_t/t10_3_runtime.py",
    "tests/test_sage_t_frame_adapters_v10_3.py",
    "tests/test_sage_t_t10_3_protocol.py",
    "tests/test_sage_t_t10_3_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_3_GOAL_PROGRESS_PILOT_PROTOCOL.md",
    "reports/SAGE_T10_3_GOAL_PROGRESS_PILOT_RUNBOOK.md",
)

canonical_json = _kernel_protocol.canonical_json
canonical_sha256 = _kernel_protocol.canonical_sha256
signed_payload = _kernel_protocol.signed_payload
write_compact_json = _kernel_protocol.write_compact_json
_read_signed_json = _kernel_protocol._read_signed_json
ManifestDriftError = _kernel_protocol.ManifestDriftError
ProtocolError = _kernel_protocol.ProtocolError


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel_protocol._repo_root()).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required artifact is absent: {path}")
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestDriftError(f"cannot authenticate input: {path}") from exc
    if not isinstance(payload, dict):
        raise ManifestDriftError(f"input is not a JSON object: {path}")
    return payload


def scientific_gates() -> dict[str, Any]:
    return {
        "qa": {
            "confident_correspondence_minimum": 0.90,
            "ambiguity_strictly_below": 0.10,
            "complete_multiframe_prefix_fraction_minimum": 0.50,
            "exact_nonidentity_transport_required": True,
            "all_comparable_exact_transports_commutative": True,
            "all_comparable_exact_transports_round_trip_exact": True,
            "goal_reachable_prevalence_minimum": 0.005,
            "goal_reachable_prevalence_maximum": 0.95,
            "goal_reachable_positive_support_minimum": 32,
            "goal_reachable_positive_games_minimum": 2,
            "positive_option_reproduced_all_panel_seeds": True,
        },
        "fit": {
            "cross_fit_auroc_minimum": 0.75,
            "brier_improvement_strictly_positive": True,
            "positive_option_maximum_rank": 8,
            "positive_option_median_rank_maximum": 4,
            "paired_binding_swap_margin_strictly_positive": True,
            "paired_option_intervention_margin_strictly_positive": True,
            "identity_only_or_no_transport_degradation_required": True,
            "identity_probe_balanced_accuracy_excess_maximum": 0.10,
            "identity_probe_chance": 1.0 / 3.0,
        },
        "confirmation": {
            "lp85_minimum_levels": 1,
            "su15_minimum_levels": 1,
            "bp35_minimum_level_margin": 0,
            "aggregate_level_advantage_minimum": 1,
            "maximum_errors": 0,
            "maximum_illegal_actions": 0,
            "game_over_rate_not_higher": True,
        },
    }


def frozen_matrix() -> dict[str, Any]:
    return {
        "games": list(SOURCE_GAMES),
        "panel": {
            "seeds": list(PANEL_SEEDS),
            "arms": list(PANEL_ARMS),
            "resets": PANEL_RESETS,
            "maximum_actions": PANEL_MAXIMUM_ACTIONS,
        },
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "controllers": list(CONFIRMATION_CONTROLLERS),
            "counterbalanced": True,
            "resets": CONFIRMATION_RESETS,
            "maximum_actions": CONFIRMATION_MAXIMUM_ACTIONS,
        },
        "maximum_actions_per_reset": MAXIMUM_ACTIONS_PER_RESET,
        "total_resets": TOTAL_RESETS,
        "total_maximum_actions": TOTAL_MAXIMUM_ACTIONS,
        "padding_authorized": False,
        "seed_replacement_authorized": False,
        "adaptive_substitution_authorized": False,
    }


def firewall_policy() -> dict[str, Any]:
    return {
        "source_games_only": list(SOURCE_GAMES),
        "source_validation_authorized": False,
        "ar25_authorized": False,
        "holdout_authorized": False,
        "production_authority": False,
        "automatic_retuning_authorized": False,
        "automatic_next_protocol_opening_authorized": False,
        "historical_t10_2_9_ledger_fit_authorized": False,
        "historical_grounded_t10_0b_actions_authorized": False,
        "physical_replay_authorized": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "output_root": DEFAULT_OUTPUT_ROOT.as_posix(),
        "offline_audit": "offline_audit.json",
        "journal": "journal",
        "checkpoint": "checkpoint.json",
        "compiled_ledger": "compiled_source_events.jsonl",
        "compact_ledger": "compact_ledger.json",
        "qa_report": "qa_report.json",
        "model_recipe": "model_recipe.json",
        "confirmation_report": "confirmation_report.json",
        "terminal_report": "t10_3_report.json",
        "write_once": True,
        "intent_before_action": True,
        "event_sealed_immediately": True,
        "branch_label_receipt_after_reset": True,
    }


def _safe_transferable_witnesses(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    witnesses: list[dict[str, Any]] = []
    for outcome in report.get("outcomes", ()):
        if not isinstance(outcome, Mapping) or outcome.get("game") not in POSITIVE_WITNESS_GAMES:
            continue
        witness = outcome.get("witness", {})
        transferable = witness.get("transferable", {}) if isinstance(witness, Mapping) else {}
        if not isinstance(transferable, Mapping):
            raise ManifestDriftError("T10.0b transferable witness is missing")
        item = {
            "macro_schema": transferable.get("macro_schema"),
            "relation": transferable.get("relation"),
            "teleological_effect": transferable.get("teleological_effect"),
            "program": transferable.get("program"),
            "program_hash": transferable.get("program_hash"),
            "steps": transferable.get("steps"),
        }
        rendered = canonical_json(item).casefold()
        if any(token in rendered for token in ('"x":', '"y":', '"grid":', '"color":', '"entity_id":')):
            raise ManifestDriftError("canonical T10.0b witness contains grounded material")
        witnesses.append(item)
    if len(witnesses) != 2:
        raise ManifestDriftError("expected exactly two positive canonical T10.0b witnesses")
    return witnesses


def build_handoff_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    terminal_path = root / T10_2_9_ROOT / "t10_2_9_report.json"
    qa_path = root / T10_2_9_ROOT / "qa_report.json"
    witness_path = root / T10_0B_REPORT
    ledger_path = root / T10_2_7_LEDGER
    terminal = _read_signed_json(terminal_path, checksum_key="terminal_checksum")
    qa = _read_signed_json(qa_path, checksum_key="report_checksum")
    witness = _read_signed_json(witness_path, checksum_key="report_checksum")
    if terminal.get("terminal_checksum") != T10_2_9_TERMINAL_CHECKSUM:
        raise ManifestDriftError("T10.2.9 terminal checksum drifted")
    if qa.get("report_checksum") != T10_2_9_QA_CHECKSUM:
        raise ManifestDriftError("T10.2.9 QA checksum drifted")
    if witness.get("report_checksum") != T10_0B_REPORT_CHECKSUM:
        raise ManifestDriftError("T10.0b positive report checksum drifted")
    if terminal.get("fit_authorized") is not False or terminal.get("physical_actions_executed") != 0:
        raise ManifestDriftError("T10.2.9 fail-closed evidence drifted")
    if qa.get("event_count") != 1370 or qa.get("passed") is not False:
        raise ManifestDriftError("T10.2.9 negative QA population drifted")
    if witness.get("status") != "PASS_T10_0_AUTHORIZE_T10_1":
        raise ManifestDriftError("T10.0b positive source status drifted")
    return signed_payload(
        {
            "format_version": HANDOFF_FORMAT_VERSION,
            "t10_2_9_terminal": artifact_descriptor(terminal_path),
            "t10_2_9_terminal_checksum": terminal["terminal_checksum"],
            "t10_2_9_qa": artifact_descriptor(qa_path),
            "t10_2_9_qa_checksum": qa["report_checksum"],
            "t10_2_9_accepted_ledger": artifact_descriptor(ledger_path),
            "t10_2_9_event_count": 1370,
            "t10_2_9_fit_excluded": True,
            "t10_0b_report": artifact_descriptor(witness_path),
            "t10_0b_report_checksum": witness["report_checksum"],
            "canonical_witnesses": _safe_transferable_witnesses(witness),
            "historical_grounded_actions_retained": False,
        },
        checksum_key="receipt_checksum",
    )


def verify_handoff_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.3 handoff receipt checksum drifted")
    if dict(receipt) != build_handoff_receipt(repo_root=repo_root):
        raise ManifestDriftError("T10.3 live handoff evidence changed")
    return {
        "handoff_verified": True,
        "t10_2_9_fit_excluded": True,
        "canonical_witness_count": len(receipt["canonical_witnesses"]),
    }


def build_manifest(
    *, repo_root: str | Path | None = None, handoff_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    verify_handoff_receipt_live(handoff_receipt, repo_root=root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel_protocol.HASH_ALGORITHM,
            "registered_phases": list(PHASES),
            "portable_code_sha256": _kernel_protocol._hash_paths(root, DEFAULT_CODE_FILES, portable=True),
            "document_sha256": _kernel_protocol._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True),
            "frozen_matrix": frozen_matrix(),
            "scientific_gates": scientific_gates(),
            "firewall": firewall_policy(),
            "artifact_contract": artifact_contract(),
            "exclusive_verdicts": list(EXCLUSIVE_VERDICTS),
            "handoff_receipt": dict(handoff_receipt),
        },
        checksum_key="manifest_checksum",
    )


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    handoff_path: str | Path = DEFAULT_HANDOFF_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repo_root)
    receipt = build_handoff_receipt(repo_root=root)
    manifest = build_manifest(repo_root=root, handoff_receipt=receipt)
    receipt_path = Path(handoff_path)
    manifest_path = Path(output_path)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if receipt_path.exists() or manifest_path.exists():
        existing = load_manifest(manifest_path, repo_root=root)
        if existing != manifest:
            raise ManifestDriftError("frozen T10.3 manifest is immutable")
        return existing
    write_compact_json(receipt_path, receipt)
    write_compact_json(manifest_path, manifest)
    return manifest


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_repository: bool = True,
    verify_live_handoff: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    manifest = _read_signed_json(source, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.3 manifest identity drifted")
    if manifest.get("frozen_matrix") != frozen_matrix():
        raise ManifestDriftError("T10.3 physical matrix drifted")
    if manifest.get("scientific_gates") != scientific_gates():
        raise ManifestDriftError("T10.3 scientific gates drifted")
    if manifest.get("firewall") != firewall_policy():
        raise ManifestDriftError("T10.3 firewall drifted")
    if manifest.get("artifact_contract") != artifact_contract():
        raise ManifestDriftError("T10.3 artifact contract drifted")
    materialized = _read_signed_json(
        root / DEFAULT_HANDOFF_RELATIVE_PATH, checksum_key="receipt_checksum"
    )
    if manifest.get("handoff_receipt") != materialized:
        raise ManifestDriftError("materialized T10.3 handoff drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel_protocol._hash_paths(root, DEFAULT_CODE_FILES, portable=True):
            raise ManifestDriftError("T10.3 code bytes drifted")
        if manifest.get("document_sha256") != _kernel_protocol._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True):
            raise ManifestDriftError("T10.3 documentation bytes drifted")
    if verify_live_handoff:
        verify_handoff_receipt_live(materialized, repo_root=root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "status"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "freeze":
            payload = freeze_manifest(output_path=args.manifest, repo_root=args.repo_root)
        else:
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_3_OFFLINE_AUDIT",
                "manifest_checksum": manifest["manifest_checksum"],
                "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
                "firewall": manifest["firewall"],
            }
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
