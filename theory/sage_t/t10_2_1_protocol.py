"""SAGE.T10.2.1 durable, time-budgeted gauge-posterior retest.

T10.2.1 changes only acquisition, persistence, and orchestration.  The frozen
T10.2 physical-event schema and scientific posterior/trainer remain the
scientific kernel.  Every active phase is separate and fail-closed; in
particular there is deliberately no ``all`` shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import t10_2_protocol as _t10_2

FORMAT_VERSION = "sage-t10.2.1-protocol-v1"
HASH_ALGORITHM = "canonical-lf-sha256-v1"
JSONL_HASH_ALGORITHM = "ordered-canonical-jsonl-root-v1"
JOURNAL_FORMAT_VERSION = "sage-t10.2.1-durable-journal-v1"
CHECKPOINT_FORMAT_VERSION = "sage-t10.2.1-collection-checkpoint-v1"
LANE_REPORT_FORMAT_VERSION = "sage-t10.2.1-lane-report-v1"
RESET_REPORT_FORMAT_VERSION = "sage-t10.2.1-reset-report-v1"
CROSS_FIT_AUDIT_FORMAT_VERSION = "sage-t10.2.1-cross-fit-audit-v1"
VALIDATION_TIMING_PROOF_FORMAT_VERSION = (
    "sage-t10.2.1-validation-timing-proof-v1"
)

PHASES = (
    "freeze",
    "collect",
    "compile",
    "replay",
    "source-train",
    "validate",
    "report",
)

IMPLEMENTATION_BASE_COMMIT = "14e476417b273fd1bec970c748dee6125ff1bc11"
PARENT_T10_1_MANIFEST_CHECKSUM = (
    "4d1c4dc8b62973187ea5e1c52e698652fdaeb424ae481b56baded0c0b2b9c1a3"
)
PARENT_T10_1_REPORT_CHECKSUM = (
    "167649e5a0e27d63668ca20ae98c57dfd50dd469204ce53a7a5af488fae6348a"
)
PARENT_T10_2_MANIFEST_CHECKSUM = (
    "f3f7a433140bb0e89ac641efc32900fc9dbbd6f701bb4b3b0cfdb193f869a8ef"
)
PARENT_T10_2_REPORT_CHECKSUM = (
    "76f9ad0ca976b32c3f36ac132a22a9c9d4984a90e9d4848c3466e1f9997410e0"
)
PARENT_T10_2_VERDICT = "DATA_OR_PROVENANCE_INVALID"

SOURCE_GAMES = _t10_2.SOURCE_GAMES
VALIDATION_GAMES = _t10_2.VALIDATION_GAMES
AR25_GAME = _t10_2.AR25_GAME
DISCOVERY_SEEDS = (101, 102, 103)
CONFIRMATION_SEEDS = (111, 112, 113)
FIT_SEED = 10_201
BOOTSTRAP_SEED = 10_202
PERMUTATION_SEED = 10_203
SOURCE_RESETS_PER_GAME_SEED = 4
SOURCE_ACTIONS_PER_RESET = 64
SOURCE_MAXIMUM_ACTIONS = 4_608
SOURCE_PRECOLLECTION_ABORTED_ACTIONS = 0
SOURCE_MAXIMUM_NEW_ACTIONS = SOURCE_MAXIMUM_ACTIONS
RESET_COOPERATIVE_STOP_SECONDS = 55.0
RESET_HARD_TIMEOUT_SECONDS = 60.0
LANE_FINALIZATION_SECONDS = 10.0
SOURCE_LANE_TIMEOUT_SECONDS = 250.0
SOURCE_STOP_NEW_ACTIONS_SECONDS = 5_100.0
SOURCE_MAXIMUM_WALL_SECONDS = 5_400.0
SOURCE_LANE_COUNT = 18
SOURCE_RESET_REPORT_COUNT = 72
REGISTERED_ACQUISITION_STOP_REASONS = frozenset(
    {
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "registered_collection_deadline",
        "resource_gate",
        "interrupted_before_reset_commit",
        "worker_exited",
        "worker_exception",
        "environment_call_unattestable",
        "parent_interrupted",
    }
)
ATTESTABLE_ACQUISITION_OR_RESOURCE_STOP_REASONS = frozenset(
    {
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "registered_collection_deadline",
        "resource_gate",
        "interrupted_before_reset_commit",
        "parent_interrupted",
    }
)

VALIDATION_SEEDS = _t10_2.VALIDATION_SEEDS
VALIDATION_RESETS_PER_GAME_SEED = _t10_2.VALIDATION_RESETS_PER_GAME_SEED
VALIDATION_ACTIONS_PER_RESET = _t10_2.VALIDATION_ACTIONS_PER_RESET
VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER = (
    _t10_2.VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
)
VALIDATION_MAXIMUM_WALL_SECONDS = _t10_2.VALIDATION_MAXIMUM_WALL_SECONDS

REGISTERED_SOURCE_CONTROLS = _t10_2.REGISTERED_SOURCE_CONTROLS
EXCLUSIVE_VERDICTS = (
    "DATA_OR_PROVENANCE_INVALID",
    "SOURCE_ACQUISITION_OR_RESOURCE_MISS",
    "MIXED_SEQUENCE_GRAMMAR_MISS",
    "FRAME_TRANSPORT_MISS",
    "GOAL_OR_DYNAMICS_MISS",
    "COMMON_POSTERIOR_MISS",
    "OPTION_SYNTHESIS_MISS",
    "SOURCE_GROUNDING_MISS",
    "SOURCE_VALIDATION_TRANSFER_MISS",
    "SAFETY_OR_RESOURCE_MISS",
    "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED",
)
SOURCE_NEGATIVE_VERDICTS = frozenset(
    {*EXCLUSIVE_VERDICTS[:8], "SAFETY_OR_RESOURCE_MISS"}
)
REGISTERED_FRAME_ORDER = _t10_2.REGISTERED_FRAME_ORDER
EVENT_FORMAT_VERSION = _t10_2.EVENT_FORMAT_VERSION
COMPACT_PROJECTION_FORMAT_VERSION = _t10_2.COMPACT_PROJECTION_FORMAT_VERSION
COMPACT_QUOTIENT_FORMAT_VERSION = _t10_2.COMPACT_QUOTIENT_FORMAT_VERSION
MAXIMUM_MODEL_VIEW_BYTES = _t10_2.MAXIMUM_MODEL_VIEW_BYTES
MAXIMUM_COMPACT_EVENT_BYTES = _t10_2.MAXIMUM_COMPACT_EVENT_BYTES

DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "theory/sage_t/sage_t10_2_1_protocol_manifest.json"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    DEFAULT_MANIFEST_RELATIVE_PATH.name
)
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "t10_2_1_durable_gauge_posterior"
)
DEFAULT_PROTOCOL_PATH = (
    Path("reports")
    / "SAGE_T10_2_1_DURABLE_TIME_BUDGETED_GAUGE_POSTERIOR_PROTOCOL.md"
)
DEFAULT_RESULT_PATH = (
    Path("reports")
    / "SAGE_T10_2_1_DURABLE_TIME_BUDGETED_GAUGE_POSTERIOR_RESULT.md"
)
CROSS_FIT_AUDIT_FILENAME = "cross_fit_audit.json"
CHALLENGER_RECIPE_FILENAME = "t10_2_1_challenger_recipe.json"
ARTIFACT_INVENTORY_FILENAME = "t10_2_1_omitted_artifacts_inventory.json"
REPORT_INVENTORY_BINDING_FILENAME = "t10_2_1_report_inventory_binding.json"
VALIDATION_OPENING_MARKER_FILENAME = "validation_opening_marker.json"
VALIDATION_OPENING_MARKER_FORMAT_VERSION = (
    "sage-t10.2.1-validation-opening-marker-v1"
)
SOURCE_FIT_OPENING_MARKER_FILENAME = "source_fit_opening_marker.json"
SOURCE_FIT_OPENING_MARKER_FORMAT_VERSION = (
    "sage-t10.2.1-source-fit-opening-marker-v1"
)

DEFAULT_SOURCE_SHARD_FILES = dict(_t10_2.DEFAULT_SOURCE_SHARD_FILES)
DEFAULT_SOURCE_METADATA_FILES = dict(_t10_2.DEFAULT_SOURCE_METADATA_FILES)
PARENT_T10_1_MANIFEST_PATH = Path(
    "theory/sage_t/sage_t10_1_source_validation_manifest.json"
)
PARENT_T10_1_REPORT_PATH = Path(
    "training/sage_t/progress_witness_v10_1/report.json"
)
PARENT_T10_2_MANIFEST_PATH = Path(
    "theory/sage_t/sage_t10_2_protocol_manifest.json"
)
PARENT_T10_2_REPORT_PATH = Path(
    "training/sage_t/t10_2_gauge_posterior/report.json"
)
FORBIDDEN_T10_2_DATA_ROOT = Path("training/sage_t/t10_2_gauge_posterior")
FORBIDDEN_T10_2_DATA_PATHS = (
    FORBIDDEN_T10_2_DATA_ROOT,
    Path("training/sage_t/t10_2_omitted_artifacts_inventory.json"),
    Path("training/sage_t/t10_2_report_inventory_binding.json"),
)

DEFAULT_CODE_FILES = (
    *_t10_2.DEFAULT_CODE_FILES,
    *_t10_2.OPTIONAL_CODE_FILES,
    "theory/sage_t/t10_2_1_protocol.py",
    "theory/sage_t/t10_2_1_runtime.py",
    "theory/sage_t/t10_2_1_artifact_inventory.py",
    "tests/test_sage_t_t10_2_1_protocol.py",
    "tests/test_sage_t_t10_2_1_runtime.py",
    "tests/test_sage_t_t10_2_1_artifact_inventory.py",
)
DEFAULT_DOCUMENTARY_INPUT_FILES = (
    ".gitattributes",
    ".gitignore",
    str(DEFAULT_PROTOCOL_PATH).replace("\\", "/"),
    str(PARENT_T10_1_MANIFEST_PATH).replace("\\", "/"),
    str(PARENT_T10_1_REPORT_PATH).replace("\\", "/"),
    str(PARENT_T10_2_MANIFEST_PATH).replace("\\", "/"),
    str(PARENT_T10_2_REPORT_PATH).replace("\\", "/"),
)
DEFAULT_PREREGISTRATION_PUBLICATION_FILES = (
    str(DEFAULT_RESULT_PATH).replace("\\", "/"),
)
DEFAULT_SOURCE_INPUT_FILES = (
    *DEFAULT_SOURCE_SHARD_FILES.values(),
    *DEFAULT_SOURCE_METADATA_FILES.values(),
)
DEFAULT_INPUT_FILES = (
    *DEFAULT_DOCUMENTARY_INPUT_FILES,
    *DEFAULT_SOURCE_INPUT_FILES,
)

ProtocolError = _t10_2.ProtocolError
ManifestDriftError = _t10_2.ManifestDriftError
FirewallError = _t10_2.FirewallError
DataGateError = _t10_2.DataGateError
GateRefusalError = _t10_2.GateRefusalError
RuntimeUnavailableError = _t10_2.RuntimeUnavailableError
ResourceGateError = _t10_2.ResourceGateError
ResourceLimits = _t10_2.ResourceLimits
ResourceSnapshot = _t10_2.ResourceSnapshot
DEFAULT_RESOURCE_LIMITS = _t10_2.DEFAULT_RESOURCE_LIMITS
GIB = _t10_2.GIB
MIB = _t10_2.MIB


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_text_bytes(raw: bytes) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestDriftError("registered text input is not UTF-8") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def canonical_file_sha256(path: str | Path) -> str:
    """Hash text portably, JSON structurally, and JSONL canonically."""

    source = Path(path)
    raw = source.read_bytes()
    suffix = source.suffix.casefold()
    if suffix == ".json":
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestDriftError(f"invalid registered JSON: {source}") from exc
        return canonical_sha256(payload)
    if suffix == ".jsonl":
        canonical_rows: list[str] = []
        for line_number, line in enumerate(_canonical_text_bytes(raw).splitlines(), 1):
            if not line.strip():
                continue
            try:
                canonical_rows.append(canonical_json(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ManifestDriftError(
                    f"invalid registered JSONL row {line_number}: {source}"
                ) from exc
        material = "".join(f"{row}\n" for row in canonical_rows).encode("utf-8")
        return hashlib.sha256(material).hexdigest()
    text_suffixes = {
        "",
        ".cfg",
        ".gitattributes",
        ".gitignore",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    material = _canonical_text_bytes(raw) if suffix in text_suffixes else raw
    return hashlib.sha256(material).hexdigest()


def ordered_jsonl_root_sha256(path: str | Path) -> str:
    """Bind ordered canonical row checksums independently of file bytes."""

    source = Path(path)
    row_hashes: list[str] = []
    for line_number, line in enumerate(
        _canonical_text_bytes(source.read_bytes()).splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row_hashes.append(canonical_sha256(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ManifestDriftError(
                f"invalid registered JSONL row {line_number}: {source}"
            ) from exc
    return canonical_sha256(
        {"algorithm": JSONL_HASH_ALGORITHM, "rows": row_hashes}
    )


def raw_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise FirewallError(f"registered path escapes repository: {candidate}")
    return resolved


def _hash_paths(
    root: Path,
    paths: Sequence[str | Path],
    *,
    portable: bool,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in paths:
        key = Path(candidate).as_posix()
        path = _resolve(root, candidate)
        if not path.is_file():
            raise ManifestDriftError(f"registered file is missing: {key}")
        result[key] = (
            canonical_file_sha256(path) if portable else raw_file_sha256(path)
        )
    return result


def _read_signed_json(path: Path, *, checksum_key: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestDriftError(f"invalid signed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ManifestDriftError(f"signed JSON must be an object: {path}")
    checksum = payload.get(checksum_key)
    unsigned = dict(payload)
    unsigned.pop(checksum_key, None)
    if checksum != canonical_sha256(unsigned):
        raise ManifestDriftError(f"signed JSON checksum drifted: {path}")
    return payload


def signed_payload(payload: Mapping[str, Any], *, checksum_key: str) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(checksum_key, None)
    return {**unsigned, checksum_key: canonical_sha256(unsigned)}


def write_compact_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _t10_2.write_compact_json(path, payload)


def _verify_parent_lineage(root: Path) -> dict[str, Any]:
    t10_1_manifest = _read_signed_json(
        _resolve(root, PARENT_T10_1_MANIFEST_PATH),
        checksum_key="manifest_checksum",
    )
    t10_1_report = _read_signed_json(
        _resolve(root, PARENT_T10_1_REPORT_PATH),
        checksum_key="report_checksum",
    )
    t10_2_manifest = _read_signed_json(
        _resolve(root, PARENT_T10_2_MANIFEST_PATH),
        checksum_key="manifest_checksum",
    )
    t10_2_report = _read_signed_json(
        _resolve(root, PARENT_T10_2_REPORT_PATH),
        checksum_key="report_checksum",
    )
    observed = {
        "t10_1_manifest_checksum": t10_1_manifest.get("manifest_checksum"),
        "t10_1_report_checksum": t10_1_report.get("report_checksum"),
        "t10_2_manifest_checksum": t10_2_manifest.get("manifest_checksum"),
        "t10_2_report_checksum": t10_2_report.get("report_checksum"),
        "t10_2_verdict": t10_2_report.get("verdict"),
    }
    expected = {
        "t10_1_manifest_checksum": PARENT_T10_1_MANIFEST_CHECKSUM,
        "t10_1_report_checksum": PARENT_T10_1_REPORT_CHECKSUM,
        "t10_2_manifest_checksum": PARENT_T10_2_MANIFEST_CHECKSUM,
        "t10_2_report_checksum": PARENT_T10_2_REPORT_CHECKSUM,
        "t10_2_verdict": PARENT_T10_2_VERDICT,
    }
    if observed != expected:
        raise ManifestDriftError("T10.1/T10.2 parent lineage drifted")
    return expected


_LINE_ENDING_AGNOSTIC_SUFFIXES = {
    "",
    ".cfg",
    ".gitattributes",
    ".gitignore",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _parent_code_faithful_digests(path: Path) -> set[str]:
    """Digests that verify byte-faithful immutable parent code.

    The parent T10.2 manifest froze ``code_sha256`` from a mixed-line-ending
    working tree: most files were LF but a handful were saved with CRLF on
    Windows, so the raw registry entry for those files is the CRLF hash.  The
    portable identity of a text file is its LF-normalized content, but a file
    that was frozen with CRLF must still verify without mutating the immutable
    parent manifest.  We therefore admit both the LF and CRLF renderings of the
    current content; the only difference tolerated is the line-ending
    convention, so the content itself is still pinned exactly.
    """

    lf_digest = canonical_file_sha256(path)
    digests = {lf_digest}
    if path.suffix.casefold() not in _LINE_ENDING_AGNOSTIC_SUFFIXES:
        return digests
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return digests
    lf = text.replace("\r\n", "\n").replace("\r", "\n")
    crlf = lf.replace("\n", "\r\n")
    digests.add(hashlib.sha256(lf.encode("utf-8")).hexdigest())
    digests.add(hashlib.sha256(crlf.encode("utf-8")).hexdigest())
    return digests


def _verify_parent_code(root: Path) -> dict[str, str]:
    parent_manifest = _read_signed_json(
        _resolve(root, PARENT_T10_2_MANIFEST_PATH),
        checksum_key="manifest_checksum",
    )
    registered = parent_manifest.get("code_sha256")
    if not isinstance(registered, Mapping) or not registered:
        raise ManifestDriftError("T10.2 parent code registry is absent")
    expected = {str(path): str(digest) for path, digest in registered.items()}
    drifted = sorted(
        path
        for path, digest in expected.items()
        if digest not in _parent_code_faithful_digests(_resolve(root, path))
    )
    if drifted:
        raise ManifestDriftError(
            "T10.2 immutable parent code drifted under LF-normalized hashing: "
            + ",".join(drifted)
        )
    return expected


def _verify_base_commit(root: Path) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{IMPLEMENTATION_BASE_COMMIT}^{{commit}}"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                IMPLEMENTATION_BASE_COMMIT,
                "HEAD",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestDriftError(
            "T10.2.1 is not descended from its registered implementation base"
        ) from exc


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestDriftError(
            f"repository proof failed: git {' '.join(arguments)}"
        ) from exc


def _verify_published_commit(root: Path, commit: str) -> None:
    refs = _git_output(
        root,
        "for-each-ref",
        "--format=%(refname)",
        f"--contains={commit}",
        "refs/remotes/origin",
    ).splitlines()
    if not any(ref.startswith("refs/remotes/origin/") for ref in refs):
        raise ManifestDriftError(
            f"registered commit is not present on origin: {commit}"
        )


def _git_text_blob_sha256(root: Path, *, commit: str, path: str) -> str:
    try:
        raw = subprocess.run(
            ["git", "show", f"{commit}:{Path(path).as_posix()}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestDriftError(
            f"preregistration blob is absent from {commit}: {path}"
        ) from exc
    return hashlib.sha256(_canonical_text_bytes(raw)).hexdigest()


def _verify_registered_paths_committed(
    root: Path, paths: Sequence[str | Path]
) -> None:
    relative_paths: list[str] = []
    for raw_path in paths:
        resolved = _resolve(root, raw_path)
        relative_paths.append(resolved.relative_to(root).as_posix())
    for relative in relative_paths:
        _git_output(root, "ls-files", "--error-unmatch", "--", relative)
    dirty = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relative_paths,
    )
    if dirty:
        raise ManifestDriftError("registered T10.2.1 files are dirty or untracked")


def _environment_metadata() -> dict[str, Any]:
    return _t10_2.environment_metadata()


def _source_plan() -> dict[str, Any]:
    return {
        "games": list(SOURCE_GAMES),
        "splits": {
            "discovery": {
                "seeds": list(DISCOVERY_SEEDS),
                "resets_per_game_seed": SOURCE_RESETS_PER_GAME_SEED,
                "maximum_actions_per_reset": SOURCE_ACTIONS_PER_RESET,
            },
            "leave_one_game_out_confirmation": {
                "seeds": list(CONFIRMATION_SEEDS),
                "resets_per_game_seed": SOURCE_RESETS_PER_GAME_SEED,
                "maximum_actions_per_reset": SOURCE_ACTIONS_PER_RESET,
            },
        },
        "fit_seed": FIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_seed": PERMUTATION_SEED,
        "maximum_actions": SOURCE_MAXIMUM_ACTIONS,
        "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
        "precollection_aborted_actions": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
        "lane_count": SOURCE_LANE_COUNT,
        "reset_report_count": SOURCE_RESET_REPORT_COUNT,
        "reset_cooperative_stop_seconds": RESET_COOPERATIVE_STOP_SECONDS,
        "reset_hard_timeout_seconds": RESET_HARD_TIMEOUT_SECONDS,
        "lane_finalization_seconds": LANE_FINALIZATION_SECONDS,
        "maximum_lane_seconds": SOURCE_LANE_TIMEOUT_SECONDS,
        "stop_new_actions_seconds": SOURCE_STOP_NEW_ACTIONS_SECONDS,
        "maximum_wall_seconds": SOURCE_MAXIMUM_WALL_SECONDS,
        "sealed_timeout_prefixes_admissible": True,
        "unresolved_intent_updates_posterior": False,
    }


def _confirmation_controller_order(seed: int) -> tuple[str, ...]:
    forward = (
        "learned",
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
    )
    return forward if int(seed) % 2 == 0 else tuple(reversed(forward))


def _validation_plan() -> dict[str, Any]:
    return {
        "games": list(VALIDATION_GAMES),
        "seeds": list(VALIDATION_SEEDS),
        "resets_per_game_seed": VALIDATION_RESETS_PER_GAME_SEED,
        "maximum_actions_per_reset": VALIDATION_ACTIONS_PER_RESET,
        "maximum_actions_per_controller": (
            VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
        ),
        "maximum_wall_seconds": VALIDATION_MAXIMUM_WALL_SECONDS,
        "counterbalanced": True,
        "posterior_reset_between_pairs": True,
        "learning_between_seeds_or_controllers": False,
    }


def _artifact_contract() -> dict[str, Any]:
    return {
        "artifact_root": DEFAULT_OUTPUT_DIR.as_posix(),
        "manifest_path": DEFAULT_MANIFEST_RELATIVE_PATH.as_posix(),
        "physical_event_format": EVENT_FORMAT_VERSION,
        "projection_format": COMPACT_PROJECTION_FORMAT_VERSION,
        "structural_quotient_format": COMPACT_QUOTIENT_FORMAT_VERSION,
        "observer_frames": list(REGISTERED_FRAME_ORDER),
        "journal_format": JOURNAL_FORMAT_VERSION,
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
        "maximum_model_view_bytes": MAXIMUM_MODEL_VIEW_BYTES,
        "maximum_compact_event_bytes": MAXIMUM_COMPACT_EVENT_BYTES,
        "raw_frames_persisted": False,
        "full_graphs_persisted": False,
    }


def _acquisition_gate() -> dict[str, Any]:
    return {
        "lane_reports": SOURCE_LANE_COUNT,
        "reset_reports": SOURCE_RESET_REPORT_COUNT,
        "logo_units": 9,
        "resets_per_confirmation_arm": 2,
        "exact_intent_accounting": True,
        "unique_event_ids": True,
        "at_least_one_sealed_event_per_confirmation_arm": True,
    }


def _frozen_firewall() -> dict[str, Any]:
    return {
        "source_train_games": list(SOURCE_GAMES),
        "source_validation_games": list(VALIDATION_GAMES),
        "source_validation_opened": False,
        "ar25_opened": False,
        "holdout_opened": False,
        "production_authority": False,
    }


def _assert_exact_manifest_tree(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    """Reject missing, extra, mistyped, or changed preregistered constants."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ManifestDriftError(
                f"T10.2.1 manifest mapping drifted: {path}"
            )
        for key, expected_value in expected.items():
            _assert_exact_manifest_tree(
                actual[key], expected_value, path=f"{path}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ManifestDriftError(
                f"T10.2.1 manifest sequence drifted: {path}"
            )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact_manifest_tree(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ManifestDriftError(f"T10.2.1 manifest constant drifted: {path}")


def build_manifest(
    *,
    repo_root: str | Path | None = None,
    code_paths: Sequence[str | Path] = DEFAULT_CODE_FILES,
    input_paths: Sequence[str | Path] = DEFAULT_INPUT_FILES,
    environment: Mapping[str, Any] | None = None,
    verify_repository: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()
    if verify_repository:
        _verify_base_commit(root)
        _verify_registered_paths_committed(
            root,
            (*code_paths, *input_paths, *DEFAULT_PREREGISTRATION_PUBLICATION_FILES),
        )
        implementation_commit = _git_output(root, "rev-parse", "HEAD")
        _verify_published_commit(root, implementation_commit)
    else:
        implementation_commit = IMPLEMENTATION_BASE_COMMIT
    parent = _verify_parent_lineage(root)
    env = dict(environment or _environment_metadata())
    parent_manifest = _read_signed_json(
        _resolve(root, PARENT_T10_2_MANIFEST_PATH),
        checksum_key="manifest_checksum",
    )
    parent_code = (
        _verify_parent_code(root)
        if verify_repository
        else {
            str(path): str(digest)
            for path, digest in parent_manifest["code_sha256"].items()
        }
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T10_2_1_COLLECTION",
        "hash_algorithm": HASH_ALGORITHM,
        "jsonl_hash_algorithm": JSONL_HASH_ALGORITHM,
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "implementation_commit": implementation_commit,
        "parent_lineage": parent,
        "parent_t10_2_code_sha256": parent_code,
        "registered_phases": list(PHASES),
        "code_sha256": _hash_paths(root, code_paths, portable=False),
        "portable_code_sha256": _hash_paths(root, code_paths, portable=True),
        "input_sha256": _hash_paths(root, input_paths, portable=False),
        "portable_input_sha256": _hash_paths(root, input_paths, portable=True),
        "input_roles": {
            "documentary_provenance_only": list(DEFAULT_DOCUMENTARY_INPUT_FILES),
            "source_replay_inputs": list(DEFAULT_SOURCE_INPUT_FILES),
            "documentary_inputs_allowed_in_model_fit": False,
        },
        "preregistration_publication": {
            "files": list(DEFAULT_PREREGISTRATION_PUBLICATION_FILES),
            "portable_initial_sha256": _hash_paths(
                root,
                DEFAULT_PREREGISTRATION_PUBLICATION_FILES,
                portable=True,
            ),
            "revalidated_as_runtime_input": False,
        },
        "environment": env,
        "environment_sha256": canonical_sha256(env),
        "frozen_source_shards": {
            game: {
                "path": path,
                "sha256": canonical_file_sha256(_resolve(root, path)),
                "ordered_root_sha256": ordered_jsonl_root_sha256(
                    _resolve(root, path)
                ),
            }
            for game, path in DEFAULT_SOURCE_SHARD_FILES.items()
        },
        "source_environment_metadata": {
            game: {
                "path": path,
                "canonical_json_sha256": canonical_file_sha256(
                    _resolve(root, path)
                ),
            }
            for game, path in DEFAULT_SOURCE_METADATA_FILES.items()
        },
        "artifact_contract": _artifact_contract(),
        "source_plan": _source_plan(),
        "validation_plan": _validation_plan(),
        "qa_gate": dict(parent_manifest["qa_gate"]),
        "source_gate": dict(parent_manifest["source_gate"]),
        "validation_gate": dict(parent_manifest["validation_gate"]),
        "acquisition_gate": _acquisition_gate(),
        "resource_limits": asdict(DEFAULT_RESOURCE_LIMITS),
        "forbidden_model_data_paths": [
            path.as_posix() for path in FORBIDDEN_T10_2_DATA_PATHS
        ],
        "firewall": _frozen_firewall(),
    }
    return signed_payload(payload, checksum_key="manifest_checksum")


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path | None = None,
    code_paths: Sequence[str | Path] = DEFAULT_CODE_FILES,
    input_paths: Sequence[str | Path] = DEFAULT_INPUT_FILES,
    environment: Mapping[str, Any] | None = None,
    verify_repository: bool = True,
) -> dict[str, Any]:
    if verify_repository:
        root = Path(repo_root or _repo_root()).resolve()
        if Path(output_path).resolve() != (root / DEFAULT_MANIFEST_RELATIVE_PATH).resolve():
            raise FirewallError("manifest must use its registered repository path")
    manifest = build_manifest(
        repo_root=repo_root,
        code_paths=code_paths,
        input_paths=input_paths,
        environment=environment,
        verify_repository=verify_repository,
    )
    write_compact_json(output_path, manifest)
    _t10_2.enforce_artifact_limit(output_path, kind="derived")
    return manifest


def _validate_manifest_constants(manifest: Mapping[str, Any]) -> None:
    expected_parent_lineage = {
        "t10_1_manifest_checksum": PARENT_T10_1_MANIFEST_CHECKSUM,
        "t10_1_report_checksum": PARENT_T10_1_REPORT_CHECKSUM,
        "t10_2_manifest_checksum": PARENT_T10_2_MANIFEST_CHECKSUM,
        "t10_2_report_checksum": PARENT_T10_2_REPORT_CHECKSUM,
        "t10_2_verdict": PARENT_T10_2_VERDICT,
    }
    preregistration = manifest.get("preregistration_publication")
    preregistration_hashes = (
        preregistration.get("portable_initial_sha256")
        if isinstance(preregistration, Mapping)
        else {}
    )
    for key, expected in (
        ("format_version", FORMAT_VERSION),
        ("status", "FROZEN_BEFORE_T10_2_1_COLLECTION"),
        ("hash_algorithm", HASH_ALGORITHM),
        ("jsonl_hash_algorithm", JSONL_HASH_ALGORITHM),
        ("implementation_base_commit", IMPLEMENTATION_BASE_COMMIT),
        ("parent_lineage", expected_parent_lineage),
        ("registered_phases", list(PHASES)),
        ("artifact_contract", _artifact_contract()),
        ("source_plan", _source_plan()),
        ("validation_plan", _validation_plan()),
        ("acquisition_gate", _acquisition_gate()),
        ("resource_limits", asdict(DEFAULT_RESOURCE_LIMITS)),
        (
            "forbidden_model_data_paths",
            [path.as_posix() for path in FORBIDDEN_T10_2_DATA_PATHS],
        ),
        (
            "input_roles",
            {
                "documentary_provenance_only": list(
                    DEFAULT_DOCUMENTARY_INPUT_FILES
                ),
                "source_replay_inputs": list(DEFAULT_SOURCE_INPUT_FILES),
                "documentary_inputs_allowed_in_model_fit": False,
            },
        ),
        (
            "preregistration_publication",
            {
                "files": list(DEFAULT_PREREGISTRATION_PUBLICATION_FILES),
                "portable_initial_sha256": preregistration_hashes,
                "revalidated_as_runtime_input": False,
            },
        ),
        ("firewall", _frozen_firewall()),
    ):
        _assert_exact_manifest_tree(manifest.get(key), expected, path=key)
    implementation_commit = str(manifest.get("implementation_commit", ""))
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ManifestDriftError("T10.2.1 implementation commit binding drifted")
    if (
        not isinstance(preregistration_hashes, Mapping)
        or set(preregistration_hashes)
        != set(DEFAULT_PREREGISTRATION_PUBLICATION_FILES)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef" for character in digest
            )
            for digest in preregistration_hashes.values()
        )
    ):
        raise ManifestDriftError(
            "T10.2.1 preregistration publication binding drifted"
        )

    shards = manifest.get("frozen_source_shards")
    metadata = manifest.get("source_environment_metadata")
    if not isinstance(shards, Mapping) or set(shards) != set(SOURCE_GAMES):
        raise ManifestDriftError("T10.2.1 frozen source shard registry drifted")
    if not isinstance(metadata, Mapping) or set(metadata) != set(SOURCE_GAMES):
        raise ManifestDriftError("T10.2.1 source metadata registry drifted")

    def is_sha256(value: Any) -> bool:
        text = str(value)
        return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

    for game in SOURCE_GAMES:
        shard = shards[game]
        source_metadata = metadata[game]
        if not isinstance(shard, Mapping) or set(shard) != {
            "path",
            "sha256",
            "ordered_root_sha256",
        }:
            raise ManifestDriftError(
                f"T10.2.1 frozen source shard schema drifted: {game}"
            )
        if (
            shard.get("path") != DEFAULT_SOURCE_SHARD_FILES[game]
            or not is_sha256(shard.get("sha256"))
            or not is_sha256(shard.get("ordered_root_sha256"))
        ):
            raise ManifestDriftError(
                f"T10.2.1 frozen source shard binding drifted: {game}"
            )
        if not isinstance(source_metadata, Mapping) or set(source_metadata) != {
            "path",
            "canonical_json_sha256",
        }:
            raise ManifestDriftError(
                f"T10.2.1 source metadata schema drifted: {game}"
            )
        if (
            source_metadata.get("path") != DEFAULT_SOURCE_METADATA_FILES[game]
            or not is_sha256(source_metadata.get("canonical_json_sha256"))
        ):
            raise ManifestDriftError(
                f"T10.2.1 source metadata binding drifted: {game}"
            )


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
    code_paths: Sequence[str | Path] = DEFAULT_CODE_FILES,
    input_paths: Sequence[str | Path] = DEFAULT_INPUT_FILES,
    environment: Mapping[str, Any] | None = None,
    verify_code: bool = True,
    verify_inputs: bool = True,
    verify_environment: bool = True,
    verify_repository: bool = True,
) -> dict[str, Any]:
    manifest = _read_signed_json(Path(path), checksum_key="manifest_checksum")
    _validate_manifest_constants(manifest)
    root = Path(repo_root or _repo_root()).resolve()
    if verify_repository:
        _verify_base_commit(root)
        if Path(path).resolve() != (root / DEFAULT_MANIFEST_RELATIVE_PATH).resolve():
            raise FirewallError("manifest escaped its registered repository path")
        implementation_commit = str(manifest["implementation_commit"])
        try:
            subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    implementation_commit,
                    "HEAD",
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ManifestDriftError(
                "implementation commit is not an ancestor of the active checkout"
            ) from exc
        _verify_published_commit(root, implementation_commit)
        preregistration_hashes = manifest["preregistration_publication"][
            "portable_initial_sha256"
        ]
        for preregistration_path, expected_digest in (
            preregistration_hashes.items()
        ):
            if (
                _git_text_blob_sha256(
                    root,
                    commit=implementation_commit,
                    path=str(preregistration_path),
                )
                != expected_digest
            ):
                raise ManifestDriftError(
                    "initial preregistration publication digest drifted: "
                    f"{preregistration_path}"
                )
        manifest_file = Path(path).resolve()
        _verify_registered_paths_committed(
            root, (*code_paths, *input_paths, manifest_file)
        )
        _verify_published_commit(root, _git_output(root, "rev-parse", "HEAD"))
    _verify_parent_lineage(root)
    parent_manifest = _read_signed_json(
        _resolve(root, PARENT_T10_2_MANIFEST_PATH),
        checksum_key="manifest_checksum",
    )
    for gate_name in ("qa_gate", "source_gate", "validation_gate"):
        _assert_exact_manifest_tree(
            manifest.get(gate_name),
            parent_manifest.get(gate_name),
            path=gate_name,
        )
    parent_code = (
        _verify_parent_code(root)
        if verify_repository
        else {
            str(path): str(digest)
            for path, digest in parent_manifest["code_sha256"].items()
        }
    )
    if manifest.get("parent_t10_2_code_sha256") != parent_code:
        raise ManifestDriftError("T10.2.1 parent code binding drifted")
    if verify_code:
        raw = _hash_paths(root, code_paths, portable=False)
        portable = _hash_paths(root, code_paths, portable=True)
        if manifest.get("portable_code_sha256") != portable:
            raise ManifestDriftError("T10.2.1 portable code hash drifted")
        # Raw hashes remain documentary for byte-level audits.  LF/CRLF-only
        # changes are accepted when the canonical portable digest is stable.
        if set(manifest.get("code_sha256", {})) != set(raw):
            raise ManifestDriftError("T10.2.1 raw code registry drifted")
    if verify_inputs:
        raw_inputs = _hash_paths(root, input_paths, portable=False)
        portable_inputs = _hash_paths(root, input_paths, portable=True)
        if manifest.get("portable_input_sha256") != portable_inputs:
            raise ManifestDriftError("T10.2.1 portable input hash drifted")
        if set(manifest.get("input_sha256", {})) != set(raw_inputs):
            raise ManifestDriftError("T10.2.1 raw input registry drifted")
        expected_shards = {
            game: {
                "path": registered_path,
                "sha256": canonical_file_sha256(
                    _resolve(root, registered_path)
                ),
                "ordered_root_sha256": ordered_jsonl_root_sha256(
                    _resolve(root, registered_path)
                ),
            }
            for game, registered_path in DEFAULT_SOURCE_SHARD_FILES.items()
        }
        if manifest.get("frozen_source_shards") != expected_shards:
            raise ManifestDriftError(
                "T10.2.1 frozen source shard binding drifted"
            )
        expected_metadata = {
            game: {
                "path": registered_path,
                "canonical_json_sha256": canonical_file_sha256(
                    _resolve(root, registered_path)
                ),
            }
            for game, registered_path in DEFAULT_SOURCE_METADATA_FILES.items()
        }
        if manifest.get("source_environment_metadata") != expected_metadata:
            raise ManifestDriftError("T10.2.1 source metadata binding drifted")
    if verify_environment:
        current = dict(environment or _environment_metadata())
        if manifest.get("environment") != current or manifest.get(
            "environment_sha256"
        ) != canonical_sha256(current):
            raise ManifestDriftError("T10.2.1 environment hash drifted")
    return manifest


def _registered_output_dir(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path | None,
) -> Path:
    root = Path(repo_root or _repo_root()).resolve()
    contract = manifest.get("artifact_contract")
    registered = (
        contract.get("artifact_root") if isinstance(contract, Mapping) else None
    )
    if registered != DEFAULT_OUTPUT_DIR.as_posix():
        raise ManifestDriftError("registered T10.2.1 artifact namespace drifted")
    candidate = Path(output_dir)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    expected = (root / DEFAULT_OUTPUT_DIR).resolve()
    if resolved != expected:
        raise FirewallError(
            "T10.2.1 output must use its registered artifact namespace"
        )
    return resolved


def _refuse_upstream_mutation_after_validation(
    *, destination: Path, phase: str
) -> None:
    if (destination / VALIDATION_OPENING_MARKER_FILENAME).exists():
        raise GateRefusalError(
            f"{phase} is frozen after the one-shot validation opening"
        )


def _refuse_upstream_mutation_after_source_fit(
    *, destination: Path, phase: str
) -> None:
    if (destination / SOURCE_FIT_OPENING_MARKER_FILENAME).exists():
        raise GateRefusalError(
            f"{phase} is frozen after the one-shot source fit opening"
        )


def enforce_environment_firewall(
    *, phase: str, game_id: str, source_gate_passed: bool = False
) -> None:
    if game_id == AR25_GAME:
        raise FirewallError("ar25 remains closed for T10.2.1")
    if phase == "collect" and game_id not in SOURCE_GAMES:
        raise FirewallError(f"non-source game blocked before collection: {game_id}")
    if phase == "validate":
        if not source_gate_passed:
            raise GateRefusalError("validation requires a signed source gate")
        if game_id not in VALIDATION_GAMES:
            raise FirewallError(f"non-validation game blocked: {game_id}")


def _factory_binding(factory: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    candidate = type(factory)
    path = Path(__file__).with_name("t10_2_1_runtime.py")
    expected = manifest.get("portable_code_sha256", {}).get(
        "theory/sage_t/t10_2_1_runtime.py"
    )
    observed = canonical_file_sha256(path) if path.is_file() else ""
    return {
        "module": candidate.__module__,
        "class": candidate.__name__,
        "source_sha256": observed,
        "manifest_checksum": str(getattr(factory, "manifest_checksum", "")),
        "code_bound": bool(
            candidate.__module__ == "theory.sage_t.t10_2_1_runtime"
            and candidate.__name__ == "T10_2_1SourceFactory"
            and observed == expected
            and getattr(factory, "manifest_checksum", "")
            == manifest.get("manifest_checksum")
        ),
    }


def _legacy_cross_fit_checks(
    *,
    manifest: Mapping[str, Any],
    source_events: Sequence[Mapping[str, Any]],
    factory: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    translated = dict(factory)
    translated.update(
        {
            "module": "theory.sage_t.t10_2_runtime",
            "class": "T10_2SourceFactory",
            "source_sha256": manifest.get("code_sha256", {}).get(
                "theory/sage_t/t10_2_runtime.py", ""
            ),
            "code_bound": True,
        }
    )
    checks = _ORIGINAL_CROSS_FIT_CHECKS(
        manifest=manifest,
        source_events=source_events,
        factory=translated,
        units=units,
    )
    checks["factory_code_bound"] = bool(
        factory.get("module") == "theory.sage_t.t10_2_1_runtime"
        and factory.get("class") == "T10_2_1SourceFactory"
        and factory.get("code_bound") is True
        and factory.get("source_sha256")
        == manifest.get("portable_code_sha256", {}).get(
            "theory/sage_t/t10_2_1_runtime.py"
        )
        and factory.get("manifest_checksum") == manifest.get("manifest_checksum")
    )
    return checks


_ORIGINAL_CROSS_FIT_CHECKS = _t10_2._cross_fit_audit_checks


def _legacy_require_source_gate(
    *,
    source_report_path: str | Path,
    manifest: Mapping[str, Any],
    output_dir: str | Path | None = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    """Verify a temporary compatibility report without opening other data."""

    del output_dir, limits
    report = _read_signed_json(
        Path(source_report_path), checksum_key="report_checksum"
    )
    checks = report.get("checks")
    controls = report.get("registered_controls")
    firewall = report.get("firewall")
    if (
        report.get("manifest_checksum") != manifest.get("manifest_checksum")
        or report.get("status") != "PASS_T10_2_SOURCE_GATE"
        or report.get("verdict") != "PASS_T10_2_SOURCE_GATE"
        or report.get("passed") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(controls, Mapping)
        or set(controls) != set(REGISTERED_SOURCE_CONTROLS)
        or not all(value is True for value in controls.values())
        or not isinstance(firewall, Mapping)
        or firewall.get("source_validation_opened") is not True
        or any(
            bool(firewall.get(key))
            for key in ("ar25_opened", "holdout_opened", "production_authority")
        )
    ):
        raise GateRefusalError("validation refused: compatibility source gate failed")
    return report


def _registered_randomness_spec() -> dict[str, Any]:
    return {
        "fit_seed": FIT_SEED,
        "fit_seed_usage": "candidate_bank_order_after_frozen_synthesis",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": 10_000,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_seed_usage": "valid_fixed_point_free_offset_order",
    }


def _seeded_permutation_pairs(
    legacy_runtime: Any,
    candidates: Sequence[Any],
    name: str,
) -> tuple[Any, str]:
    """Run only the transport permutation in a registered seeded order."""

    if name != "deterministically_permuted_transport":
        return legacy_runtime.__t10_2_1_original_ablation_pairs(candidates, name)
    bank = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, legacy_runtime.JointGaugeHypothesis)
            ),
            key=lambda candidate: candidate.canonical_hash,
        )
    )[:64]
    if len(bank) != min(64, len(candidates)):
        return (), "non_gauge_candidate"
    if len({candidate.canonical_hash for candidate in bank}) != len(bank):
        return (), "reference_capacity_collapse"
    if len(bank) < 2:
        return (), "fixed_point_free_permutation_requires_two_candidates"
    source_tokens = tuple(
        legacy_runtime._stable_hash(
            legacy_runtime._ablation_component(candidate, name)
        )
        for candidate in bank
    )
    offsets = list(range(1, len(bank)))
    random.Random(PERMUTATION_SEED).shuffle(offsets)
    for offset in offsets:
        donors = bank[offset:] + bank[:offset]
        donor_tokens = tuple(
            legacy_runtime._stable_hash(
                legacy_runtime._ablation_component(donor, name)
            )
            for donor in donors
        )
        if any(
            source == donor
            for source, donor in zip(source_tokens, donor_tokens, strict=True)
        ) or Counter(source_tokens) != Counter(donor_tokens):
            continue
        try:
            alternatives = tuple(
                legacy_runtime._replace_ablation_component(candidate, donor, name)
                for candidate, donor in zip(bank, donors, strict=True)
            )
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            candidate.canonical_hash == altered.canonical_hash
            for candidate, altered in zip(bank, alternatives, strict=True)
        ) or len({candidate.canonical_hash for candidate in alternatives}) != len(
            alternatives
        ):
            continue
        return tuple(zip(bank, alternatives, strict=True)), ""
    return (), "no_valid_fixed_point_free_component_permutation"


@contextmanager
def _legacy_bindings() -> Any:
    """Version-bind the frozen T10.2 scientific kernel without editing it."""

    from . import t10_2_runtime as legacy_runtime

    original_bootstrap = legacy_runtime.paired_bootstrap_lower
    original_synthesize = legacy_runtime._synthesize_gauge_candidates
    original_ablation_pairs = legacy_runtime._ablation_pairs
    original_source_trainer = legacy_runtime.run_source_trainer
    original_rebuild = legacy_runtime._rebuild_challenger_posterior

    def registered_bootstrap(
        values: Sequence[float],
        *,
        samples: int = 10_000,
        seed: int = BOOTSTRAP_SEED,
    ) -> float:
        if samples != 10_000 or seed != BOOTSTRAP_SEED:
            raise DataGateError("paired bootstrap randomness drifted")
        return original_bootstrap(values, samples=samples, seed=seed)

    def registered_synthesize(
        events: Sequence[Mapping[str, Any]], *, maximum: int = 256
    ) -> tuple[Any, ...]:
        candidates = list(original_synthesize(events, maximum=maximum))
        random.Random(FIT_SEED).shuffle(candidates)
        return tuple(candidates)

    def registered_code_binding(manifest: Mapping[str, Any]) -> dict[str, str]:
        paths = (
            *legacy_runtime._CHALLENGER_CODE_PATHS,
            "theory/sage_t/t10_2_1_protocol.py",
            "theory/sage_t/t10_2_1_runtime.py",
        )
        portable = manifest.get("portable_code_sha256")
        if not isinstance(portable, Mapping):
            raise ManifestDriftError("portable challenger code binding is absent")
        binding: dict[str, str] = {}
        for path in paths:
            expected = portable.get(path)
            observed = canonical_file_sha256(_repo_root() / path)
            if not isinstance(expected, str) or observed != expected:
                raise ManifestDriftError(f"challenger code drifted: {path}")
            binding[path] = expected
        return binding

    def registered_timing_code_binding(
        manifest: Mapping[str, Any],
    ) -> dict[str, str]:
        paths = (
            *_t10_2.VALIDATION_TIMING_CODE_PATHS,
            "theory/sage_t/t10_2_1_protocol.py",
            "theory/sage_t/t10_2_1_runtime.py",
        )
        portable = manifest.get("portable_code_sha256")
        if not isinstance(portable, Mapping):
            raise ManifestDriftError("portable timing code binding is absent")
        binding: dict[str, str] = {}
        for path in paths:
            expected = portable.get(path)
            if (
                not isinstance(expected, str)
                or canonical_file_sha256(_repo_root() / path) != expected
            ):
                raise ManifestDriftError(f"validation timing code drifted: {path}")
            binding[path] = expected
        return binding

    def registered_rebuild(**kwargs: Any) -> Any:
        recipe = kwargs.get("recipe")
        if not isinstance(recipe, Mapping) or recipe.get(
            "registered_randomness"
        ) != _registered_randomness_spec():
            raise ManifestDriftError("challenger randomness binding drifted")
        return original_rebuild(**kwargs)

    def registered_source_trainer(**kwargs: Any) -> dict[str, Any]:
        metrics = dict(original_source_trainer(**kwargs))
        metrics["registered_randomness"] = _registered_randomness_spec()
        binding = metrics.get("challenger_recipe")
        if isinstance(binding, Mapping) and binding.get("bound") is True:
            destination = Path(kwargs["output_dir"])
            recipe_path = destination / CHALLENGER_RECIPE_FILENAME
            recipe = _read_signed_json(
                recipe_path, checksum_key="recipe_checksum"
            )
            recipe["registered_randomness"] = _registered_randomness_spec()
            recipe = signed_payload(recipe, checksum_key="recipe_checksum")
            write_compact_json(recipe_path, recipe)
            metrics["challenger_recipe"] = {
                **dict(binding),
                "artifact": _t10_2.artifact_descriptor(recipe_path),
                "recipe_checksum": recipe["recipe_checksum"],
            }
        return metrics

    def registered_trainer_binding(
        trainer: Callable[..., Any],
        *,
        manifest: Mapping[str, Any],
        repo_root: Path,
    ) -> bool:
        if trainer is not registered_source_trainer:
            return False
        portable = manifest.get("portable_code_sha256")
        return bool(
            isinstance(portable, Mapping)
            and all(
                canonical_file_sha256(repo_root / path) == portable.get(path)
                for path in (
                    "theory/sage_t/t10_2_1_protocol.py",
                    "theory/sage_t/t10_2_1_runtime.py",
                )
            )
        )

    setattr(
        legacy_runtime,
        "__t10_2_1_original_ablation_pairs",
        original_ablation_pairs,
    )
    protocol_overrides = {
        "FORMAT_VERSION": FORMAT_VERSION,
        "DISCOVERY_SEEDS": DISCOVERY_SEEDS,
        "CONFIRMATION_SEEDS": CONFIRMATION_SEEDS,
        "SOURCE_RESETS_PER_GAME_SEED": SOURCE_RESETS_PER_GAME_SEED,
        "SOURCE_ACTIONS_PER_RESET": SOURCE_ACTIONS_PER_RESET,
        "SOURCE_MAXIMUM_ACTIONS": SOURCE_MAXIMUM_ACTIONS,
        "SOURCE_PRECOLLECTION_ABORTED_ACTIONS": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
        "SOURCE_MAXIMUM_NEW_ACTIONS": SOURCE_MAXIMUM_NEW_ACTIONS,
        "SOURCE_MAXIMUM_WALL_SECONDS": SOURCE_MAXIMUM_WALL_SECONDS,
        "DEFAULT_MANIFEST_PATH": DEFAULT_MANIFEST_PATH,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "CHALLENGER_RECIPE_FILENAME": CHALLENGER_RECIPE_FILENAME,
        "CROSS_FIT_AUDIT_FILENAME": CROSS_FIT_AUDIT_FILENAME,
        "CROSS_FIT_AUDIT_FORMAT_VERSION": CROSS_FIT_AUDIT_FORMAT_VERSION,
        "VALIDATION_TIMING_PROOF_FORMAT_VERSION": (
            VALIDATION_TIMING_PROOF_FORMAT_VERSION
        ),
        "load_manifest": load_manifest,
        "require_source_gate": _legacy_require_source_gate,
        "_cross_fit_audit_checks": _legacy_cross_fit_checks,
        "_expected_cross_fit_resets": _confirmation_controller_order,
        "_code_bound_source_trainer": registered_trainer_binding,
        "_validation_timing_code_binding": registered_timing_code_binding,
        "paired_bootstrap_lower": registered_bootstrap,
    }
    runtime_overrides = {
        "DISCOVERY_SEEDS": DISCOVERY_SEEDS,
        "CONFIRMATION_SEEDS": CONFIRMATION_SEEDS,
        "SOURCE_RESETS_PER_GAME_SEED": SOURCE_RESETS_PER_GAME_SEED,
        "SOURCE_ACTIONS_PER_RESET": SOURCE_ACTIONS_PER_RESET,
        "VALIDATION_SEEDS": VALIDATION_SEEDS,
        "CHALLENGER_RECIPE_FILENAME": CHALLENGER_RECIPE_FILENAME,
        "CROSS_FIT_AUDIT_FILENAME": CROSS_FIT_AUDIT_FILENAME,
        "paired_bootstrap_lower": registered_bootstrap,
        "_synthesize_gauge_candidates": registered_synthesize,
        "_ablation_pairs": lambda candidates, name: _seeded_permutation_pairs(
            legacy_runtime, candidates, name
        ),
        "_challenger_code_binding": registered_code_binding,
        "_rebuild_challenger_posterior": registered_rebuild,
        "run_source_trainer": registered_source_trainer,
        "file_sha256": canonical_file_sha256,
    }
    saved_protocol = {
        key: getattr(_t10_2, key) for key in protocol_overrides
    }
    saved_runtime = {
        key: getattr(legacy_runtime, key) for key in runtime_overrides
    }
    try:
        for key, value in protocol_overrides.items():
            setattr(_t10_2, key, value)
        for key, value in runtime_overrides.items():
            setattr(legacy_runtime, key, value)
        yield legacy_runtime
    finally:
        for key, value in saved_runtime.items():
            setattr(legacy_runtime, key, value)
        for key, value in saved_protocol.items():
            setattr(_t10_2, key, value)
        delattr(legacy_runtime, "__t10_2_1_original_ablation_pairs")


def _build_acquisition_failure_report(
    *,
    manifest: Mapping[str, Any],
    collection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    collection_status = str(collection.get("status", ""))
    verdict = (
        "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
        if (
            collection_status == "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
            and _collection_failure_is_attested(
                collection, manifest=manifest, output_dir=output_dir
            )
        )
        else "DATA_OR_PROVENANCE_INVALID"
    )
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "phase": "compile",
            "status": verdict,
            "verdict": verdict,
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": False,
            "checks": {
                "acquisition_gate_passed": False,
                "compiler_invoked": False,
                "trainer_invoked": False,
                "validation_closed": True,
            },
            "collection_report_checksum": collection.get("report_checksum"),
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )


def _acquisition_failure_report(
    *,
    manifest: Mapping[str, Any],
    collection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    payload = _build_acquisition_failure_report(
        manifest=manifest,
        collection=collection,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "compile_report.json"
    if report_path.exists():
        existing = _read_signed_json(
            report_path, checksum_key="report_checksum"
        )
        if canonical_json(existing) != canonical_json(payload):
            raise ManifestDriftError(
                "existing terminal compile report did not reconstruct"
            )
        return existing
    write_compact_json(report_path, payload)
    return payload


def _collection_acquisition_passed(
    collection: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
) -> bool:
    checks = collection.get("checks")
    accounting = collection.get("action_accounting")
    durability = collection.get("durability")
    cross_fit_checks = collection.get("cross_fit_checks")
    firewall = collection.get("firewall")
    invocation = collection.get("invocation")
    event_count = collection.get("event_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (
            event_count,
            accounting.get("authorized_intent_count")
            if isinstance(accounting, Mapping)
            else None,
            accounting.get("sealed_event_count")
            if isinstance(accounting, Mapping)
            else None,
            accounting.get("explicitly_unresolved_intent_count")
            if isinstance(accounting, Mapping)
            else None,
            accounting.get("unknown_intent_count")
            if isinstance(accounting, Mapping)
            else None,
            accounting.get("maximum_authorized_intents")
            if isinstance(accounting, Mapping)
            else None,
            durability.get("lane_report_count")
            if isinstance(durability, Mapping)
            else None,
            durability.get("reset_report_count")
            if isinstance(durability, Mapping)
            else None,
            durability.get("physical_steps_replayed_on_resume")
            if isinstance(durability, Mapping)
            else None,
        )
    ):
        return False
    authorized = int(accounting["authorized_intent_count"])
    sealed = int(accounting["sealed_event_count"])
    unresolved = int(accounting["explicitly_unresolved_intent_count"])
    unknown = int(accounting["unknown_intent_count"])
    maximum = accounting.get("maximum_authorized_intents")
    passed = bool(
        collection.get("status") == "T10_2_1_SOURCE_COLLECTION_COMPLETE"
        and isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(cross_fit_checks, Mapping)
        and cross_fit_checks
        and all(value is True for value in cross_fit_checks.values())
        and isinstance(durability, Mapping)
        and durability.get("lane_report_count") == SOURCE_LANE_COUNT
        and durability.get("reset_report_count") == SOURCE_RESET_REPORT_COUNT
        and durability.get("journal_reconstructed") is True
        and durability.get("checkpoint_reconstructed") is True
        and durability.get("physical_steps_replayed_on_resume") == 0
        and accounting.get("equation_holds") is True
        and maximum == SOURCE_MAXIMUM_ACTIONS
        and authorized == sealed + unresolved
        and unresolved == 0
        and unknown == 0
        and sealed == event_count
        and 0 < authorized <= SOURCE_MAXIMUM_ACTIONS
        and isinstance(invocation, Mapping)
        and invocation.get("terminal_status") == "CLOSED"
        and invocation.get("absolute_wall_bound") is True
        and isinstance(invocation.get("open_state_checksum"), str)
        and len(str(invocation.get("open_state_checksum"))) == 64
        and isinstance(invocation.get("terminal_checksum"), str)
        and len(str(invocation.get("terminal_checksum"))) == 64
        and isinstance(invocation.get("report_core_checksum"), str)
        and len(str(invocation.get("report_core_checksum"))) == 64
        and isinstance(firewall, Mapping)
        and not any(
            bool(firewall.get(key))
            for key in (
                "source_validation_opened",
                "ar25_opened",
                "holdout_opened",
                "production_authority",
            )
        )
    )
    return bool(
        passed
        and (
            output_dir is None
            or (
                manifest is not None
                and _collection_durable_evidence_matches(
                    collection, manifest=manifest, output_dir=output_dir
                )
            )
        )
    )


def _collection_failure_is_attested(
    collection: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    output_dir: Path | None = None,
) -> bool:
    accounting = collection.get("action_accounting")
    durability = collection.get("durability")
    firewall = collection.get("firewall")
    invocation = collection.get("invocation")
    event_count = collection.get("event_count")
    if not isinstance(accounting, Mapping) or not isinstance(durability, Mapping):
        return False
    integer_values = (
        event_count,
        accounting.get("authorized_intent_count"),
        accounting.get("sealed_event_count"),
        accounting.get("explicitly_unresolved_intent_count"),
        accounting.get("unknown_intent_count"),
        durability.get("lane_report_count"),
        durability.get("reset_report_count"),
        durability.get("physical_steps_replayed_on_resume"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        return False
    (
        observed_events,
        authorized,
        sealed,
        unresolved,
        unknown,
        lane_reports,
        reset_reports,
        replayed_steps,
    ) = (int(value) for value in integer_values)
    checks = collection.get("checks")
    timing = collection.get("timing")
    terminal_reason = str(collection.get("terminal_reason", ""))
    registered_stop = (
        terminal_reason in ATTESTABLE_ACQUISITION_OR_RESOURCE_STOP_REASONS
    )
    finalization_timeout = bool(
        terminal_reason == "registered_collection_deadline"
        and (
            not isinstance(invocation, Mapping)
            or invocation.get("terminal_status") == "HARD_TIMEOUT"
        )
    )
    incomplete_or_finalization_timeout = bool(
        lane_reports < SOURCE_LANE_COUNT
        or reset_reports < SOURCE_RESET_REPORT_COUNT
        or unresolved > 0
        or finalization_timeout
    )
    required_true_checks = (
        "action_equation_holds",
        "no_unknown_intents",
        "authorized_action_cap",
        "sealed_events_bound",
        "journal_reconstructed",
        "checkpoint_reconstructed",
        "physical_steps_not_replayed",
        "absolute_wall_bound",
        "source_firewall_closed",
    )
    attested = bool(
        accounting.get("maximum_authorized_intents") == SOURCE_MAXIMUM_ACTIONS
        and accounting.get("equation_holds") is True
        and authorized == sealed + unresolved
        and observed_events == sealed
        and unknown == 0
        and 0 <= authorized <= SOURCE_MAXIMUM_ACTIONS
        and sealed >= 0
        and unresolved >= 0
        and 0 <= lane_reports <= SOURCE_LANE_COUNT
        and 0 <= reset_reports <= SOURCE_RESET_REPORT_COUNT
        and durability.get("journal_reconstructed") is True
        and durability.get("checkpoint_reconstructed") is True
        and replayed_steps == 0
        and incomplete_or_finalization_timeout
        and registered_stop
        and isinstance(checks, Mapping)
        and all(checks.get(name) is True for name in required_true_checks)
        and isinstance(timing, Mapping)
        and timing.get("stop_new_actions_seconds")
        == SOURCE_STOP_NEW_ACTIONS_SECONDS
        and timing.get("absolute_seconds") == SOURCE_MAXIMUM_WALL_SECONDS
        and isinstance(firewall, Mapping)
        and not any(
            bool(firewall.get(key))
            for key in (
                "source_validation_opened",
                "ar25_opened",
                "holdout_opened",
                "production_authority",
            )
        )
    )
    if not attested or output_dir is None:
        return attested
    return bool(
        manifest is not None
        and _collection_durable_evidence_matches(
            collection, manifest=manifest, output_dir=output_dir
        )
    )


def _collection_durable_evidence_matches(
    collection: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    output_dir: Path,
    require_invocation: bool = True,
) -> bool:
    try:
        from .t10_2_1_runtime import (
            INVOCATION_STATE_FILENAME,
            INVOCATION_TERMINAL_FILENAME,
            JOURNAL_DIRECTORY_NAME,
            DurableCollectionJournal,
            _read_invocation_state,
            _read_invocation_terminal,
            _validate_report_invocation_binding,
        )

        journal_metadata_path = output_dir / JOURNAL_DIRECTORY_NAME / "journal.json"
        checkpoint_path = output_dir / "source_collection_checkpoint.json"
        if not journal_metadata_path.is_file() or not checkpoint_path.is_file():
            return False
        journal = DurableCollectionJournal(
            output_dir / JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(collection.get("manifest_checksum", "")),
        )
        reconstructed_accounting = journal.accounting().to_dict()
        checkpoint = journal.load_checkpoint()
        if checkpoint is None:
            return False
        lane_reports = journal.lane_reports()
        events_path = output_dir / "source_events.jsonl"
        cross_fit_path = output_dir / CROSS_FIT_AUDIT_FILENAME
        cross_fit = _read_signed_json(
            cross_fit_path, checksum_key="audit_checksum"
        )
        events = _t10_2.read_event_ledger(events_path)
        journal_events = journal.all_events(complete_resets_only=False)
        durability = collection.get("durability")
        accounting = collection.get("action_accounting")
        observed_events = collection.get("event_count")
        if not isinstance(durability, Mapping) or not isinstance(accounting, Mapping):
            return False
        invocation_matches = True
        if require_invocation:
            opened = _read_invocation_state(
                output_dir / INVOCATION_STATE_FILENAME
            )
            terminal = _read_invocation_terminal(
                output_dir / INVOCATION_TERMINAL_FILENAME,
                opened=opened,
            )
            _validate_report_invocation_binding(
                collection,
                opened=opened,
                terminal=terminal,
                required=True,
            )
            invocation_matches = bool(
                not (
                    collection.get("status")
                    == "T10_2_1_SOURCE_COLLECTION_COMPLETE"
                    and terminal is not None
                    and terminal.get("status") != "CLOSED"
                )
            )
        if isinstance(observed_events, bool) or not isinstance(observed_events, int):
            return False
        units = cross_fit.get("units")
        factory = cross_fit.get("factory")
        if not isinstance(units, list) or not isinstance(factory, Mapping):
            return False
        with _legacy_bindings():
            canonical_units = _t10_2._canonical_cross_fit_units(units)
            reconstructed_cross_fit_checks = _legacy_cross_fit_checks(
                manifest=manifest,
                source_events=events,
                factory=factory,
                units=units,
            )
        return bool(
            invocation_matches
            and reconstructed_accounting == dict(accounting)
            and collection.get("manifest_checksum")
            == manifest.get("manifest_checksum")
            and tuple(events) == tuple(journal_events)
            and collection.get("events") == _t10_2.artifact_descriptor(events_path)
            and collection.get("cross_fit_audit")
            == _t10_2.artifact_descriptor(cross_fit_path)
            and len(events) == observed_events
            and cross_fit.get("manifest_checksum")
            == collection.get("manifest_checksum")
            and cross_fit.get("format_version")
            == CROSS_FIT_AUDIT_FORMAT_VERSION
            and cross_fit.get("source_events")
            == _t10_2.artifact_descriptor(events_path)
            and cross_fit.get("source_event_ids_sha256")
            == canonical_sha256(
                [str(event.get("event_id", "")) for event in events]
            )
            and cross_fit.get("registered_unit_count") == len(units)
            and units == canonical_units
            and set(factory)
            == {
                "module",
                "class",
                "source_sha256",
                "manifest_checksum",
                "code_bound",
            }
            and cross_fit.get("checks") == reconstructed_cross_fit_checks
            and cross_fit.get("passed")
            is all(reconstructed_cross_fit_checks.values())
            and collection.get("cross_fit_checks")
            == reconstructed_cross_fit_checks
            and collection.get("factory") == factory
            and durability.get("journal_metadata")
            == _t10_2.artifact_descriptor(journal_metadata_path)
            and durability.get("checkpoint")
            == _t10_2.artifact_descriptor(checkpoint_path)
            and durability.get("checkpoint_checksum")
            == checkpoint.checkpoint_checksum
            and checkpoint.lane_reports == lane_reports
            and checkpoint.journal_reconstructed is True
            and checkpoint.checkpoint_reconstructed is True
            and checkpoint.physical_steps_replayed_on_resume == 0
        )
    except (OSError, ProtocolError, RuntimeError, ValueError, KeyError):
        return False


def compile_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    _refuse_upstream_mutation_after_validation(
        destination=destination, phase="compile"
    )
    _refuse_upstream_mutation_after_source_fit(
        destination=destination, phase="compile"
    )
    collection = _read_signed_json(
        destination / "collection_report.json", checksum_key="report_checksum"
    )
    if collection.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("collection/manifest binding drifted")
    if not _collection_acquisition_passed(
        collection, manifest=manifest, output_dir=destination
    ):
        return _acquisition_failure_report(
            manifest=manifest,
            collection=collection,
            output_dir=destination,
        )
    event_count = int(collection["event_count"])
    compatibility = dict(collection)
    compatibility.update(
        {
            "status": "T10_2_SOURCE_COLLECTION_COMPLETE",
            "precollection_aborted_actions": 0,
            "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
            # The frozen compiler validates the physical ledger.  T10.2.1
            # separately binds all authorized and unresolved intents in the
            # durable acquisition report.
            "accounted_action_count": event_count,
        }
    )
    compatibility = signed_payload(
        compatibility, checksum_key="report_checksum"
    )
    with tempfile.TemporaryDirectory(prefix="sage-t10-2-1-compile-") as temp:
        compatibility_path = Path(temp) / "collection_report.json"
        write_compact_json(compatibility_path, compatibility)
        with _legacy_bindings():
            report = _t10_2.compile_phase(
                manifest_path=manifest_path,
                output_dir=destination,
                collection_report_path=compatibility_path,
                repo_root=repo_root,
            )
    report = dict(report)
    report["status"] = {
        "T10_2_FRESH_INTEGRITY_COMPLETE": "T10_2_1_FRESH_INTEGRITY_COMPLETE",
        "PASS_T10_2_QA": "PASS_T10_2_1_QA",
    }.get(str(report.get("status")), report.get("status"))
    report["collection_report_checksum"] = collection["report_checksum"]
    report = signed_payload(report, checksum_key="report_checksum")
    write_compact_json(destination / "compile_report.json", report)
    return report


def replay_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    _refuse_upstream_mutation_after_validation(
        destination=destination, phase="replay"
    )
    _refuse_upstream_mutation_after_source_fit(
        destination=destination, phase="replay"
    )
    compile_report = _read_signed_json(
        destination / "compile_report.json", checksum_key="report_checksum"
    )
    if compile_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("compile/manifest binding drifted")
    compile_status = compile_report.get("status")
    compile_checks = compile_report.get("checks")
    compile_passed = bool(
        compile_status
        in {"T10_2_1_FRESH_INTEGRITY_COMPLETE", "PASS_T10_2_1_QA"}
        and compile_report.get("passed") is True
        and isinstance(compile_checks, Mapping)
        and compile_checks
        and all(value is True for value in compile_checks.values())
        and (
            compile_status == "PASS_T10_2_1_QA"
            or compile_report.get("integrity_passed") is True
        )
    )
    if not compile_passed:
        raise GateRefusalError("replay blocked: compile integrity did not pass")
    root = Path(repo_root or _repo_root()).resolve()
    compatibility = dict(compile_report)
    compatibility["status"] = {
        "T10_2_1_FRESH_INTEGRITY_COMPLETE": "T10_2_FRESH_INTEGRITY_COMPLETE",
        "PASS_T10_2_1_QA": "PASS_T10_2_QA",
    }.get(str(compatibility.get("status")), compatibility.get("status"))
    compatibility = signed_payload(
        compatibility, checksum_key="report_checksum"
    )
    with _legacy_bindings() as legacy_runtime:
        with tempfile.TemporaryDirectory(prefix="sage-t10-2-1-replay-") as temp:
            replay_input = Path(temp) / "manifest_bound_replay.jsonl"
            compile_compatibility_path = Path(temp) / "compile_report.json"
            write_compact_json(compile_compatibility_path, compatibility)
            legacy_runtime.build_v4_3_replay_ledger(
                manifest=manifest,
                repo_root=root,
                output_path=replay_input,
            )
            report = _t10_2.replay_phase(
                replay_input_path=replay_input,
                manifest_path=manifest_path,
                output_dir=destination,
                compile_report_path=compile_compatibility_path,
                repo_root=root,
            )
    report = dict(report)
    report["status"] = "T10_2_1_SOURCE_REPLAY_COMPLETE"
    report["compile_report_checksum"] = compile_report["report_checksum"]
    report["input"] = {
        "kind": "manifest_bound_v4_3_source_shards",
        "shards": dict(manifest["frozen_source_shards"]),
        "conversion_kernel": "theory/sage_t/t10_2_runtime.py",
    }
    report = signed_payload(report, checksum_key="report_checksum")
    write_compact_json(destination / "replay_report.json", report)
    return report


def _build_terminal_source_failure_report(
    *,
    manifest: Mapping[str, Any],
    compile_report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    verdict = (
        "DATA_OR_PROVENANCE_INVALID"
        if compile_report.get("status") == "DATA_OR_PROVENANCE_INVALID"
        else "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
    )
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "phase": "source-train",
            "status": verdict,
            "verdict": verdict,
            "manifest_checksum": manifest["manifest_checksum"],
            "terminal_stage": "acquisition",
            "reason": "source evidence failed before replay or fit",
            "checks": {
                "acquisition_gate_passed": False,
                "trainer_invoked": False,
                "source_validation_closed": True,
                "ar25_closed": True,
                "holdout_closed": True,
            },
            "compile_report_checksum": compile_report.get("report_checksum"),
            "inputs": {
                "compile_report": _t10_2.artifact_descriptor(
                    output_dir / "compile_report.json"
                )
            },
            "registered_controls": {
                name: False for name in REGISTERED_SOURCE_CONTROLS
            },
            "passed": False,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )


def _terminal_source_failure_report(
    *,
    manifest: Mapping[str, Any],
    compile_report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    report = _build_terminal_source_failure_report(
        manifest=manifest,
        compile_report=compile_report,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "source_report.json"
    if report_path.exists():
        existing = _read_signed_json(
            report_path, checksum_key="report_checksum"
        )
        if canonical_json(existing) != canonical_json(report):
            raise ManifestDriftError(
                "existing terminal source report did not reconstruct"
            )
        return existing
    write_compact_json(report_path, report)
    return report


def _source_fit_opening_payload(
    *,
    manifest: Mapping[str, Any],
    destination: Path,
    compile_report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_paths = {
        "compile_report": destination / "compile_report.json",
        "replay_report": destination / "replay_report.json",
        "fresh_events": destination / "source_events.jsonl",
        "replay_events": destination / "replay_events.jsonl",
        "cross_fit_audit": destination / CROSS_FIT_AUDIT_FILENAME,
    }
    evidence = {
        key: _t10_2.artifact_descriptor(path)
        for key, path in evidence_paths.items()
    }
    return signed_payload(
        {
            "format_version": SOURCE_FIT_OPENING_MARKER_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "compile_report_checksum": compile_report["report_checksum"],
            "replay_report_checksum": replay_report["report_checksum"],
            "evidence": evidence,
            "evidence_sha256": canonical_sha256(evidence),
            "opened_once": True,
        },
        checksum_key="opening_checksum",
    )


def _verify_source_fit_opening_marker(
    *,
    path: Path,
    manifest: Mapping[str, Any],
    destination: Path,
    compile_report: Mapping[str, Any],
    replay_report: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _read_signed_json(path, checksum_key="opening_checksum")
    expected = _source_fit_opening_payload(
        manifest=manifest,
        destination=destination,
        compile_report=compile_report,
        replay_report=replay_report,
    )
    if marker != expected:
        raise ManifestDriftError("source-fit opening marker drifted")
    return marker


def _reconstruct_existing_source_report(
    *,
    manifest: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    source_path = destination / "source_report.json"
    source = _read_signed_json(source_path, checksum_key="report_checksum")
    if source.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise ManifestDriftError("source report/manifest binding drifted")
    if source.get("status") == "PASS_T10_2_1_SOURCE_GATE":
        return _require_source_gate(
            manifest=manifest,
            source_report_path=source_path,
            output_dir=destination,
        )
    _reconstruct_negative_source_report(
        manifest=manifest,
        source=source,
        destination=destination,
    )
    return source


def source_train_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    _refuse_upstream_mutation_after_validation(
        destination=destination, phase="source-train"
    )
    compile_report = _read_signed_json(
        destination / "compile_report.json", checksum_key="report_checksum"
    )
    if compile_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("compile/manifest binding drifted")
    if compile_report.get("status") in {
        "SOURCE_ACQUISITION_OR_RESOURCE_MISS",
        "DATA_OR_PROVENANCE_INVALID",
    }:
        return _terminal_source_failure_report(
            manifest=manifest,
            compile_report=compile_report,
            output_dir=destination,
        )
    compile_checks = compile_report.get("checks")
    compile_status = compile_report.get("status")
    if not (
        compile_status
        in {"T10_2_1_FRESH_INTEGRITY_COMPLETE", "PASS_T10_2_1_QA"}
        and compile_report.get("passed") is True
        and isinstance(compile_checks, Mapping)
        and compile_checks
        and all(value is True for value in compile_checks.values())
        and (
            compile_status == "PASS_T10_2_1_QA"
            or compile_report.get("integrity_passed") is True
        )
    ):
        raise GateRefusalError("source training requires passing compile integrity")
    replay_report = _read_signed_json(
        destination / "replay_report.json", checksum_key="report_checksum"
    )
    if not (
        replay_report.get("manifest_checksum") == manifest["manifest_checksum"]
        and replay_report.get("status") == "T10_2_1_SOURCE_REPLAY_COMPLETE"
        and replay_report.get("compile_report_checksum")
        == compile_report.get("report_checksum")
    ):
        raise ManifestDriftError("replay/compile/manifest binding drifted")
    source_fit_marker = destination / SOURCE_FIT_OPENING_MARKER_FILENAME
    source_path = destination / "source_report.json"
    if source_fit_marker.exists():
        _verify_source_fit_opening_marker(
            path=source_fit_marker,
            manifest=manifest,
            destination=destination,
            compile_report=compile_report,
            replay_report=replay_report,
        )
        if source_path.is_file():
            return _reconstruct_existing_source_report(
                manifest=manifest,
                destination=destination,
            )
        raise GateRefusalError(
            "source fit was opened but lacks a reconstructible terminal report"
        )
    if source_path.exists() or (destination / CHALLENGER_RECIPE_FILENAME).exists():
        raise ManifestDriftError(
            "source-fit artifacts exist without their one-shot opening marker"
        )
    _create_one_shot_marker(
        source_fit_marker,
        _source_fit_opening_payload(
            manifest=manifest,
            destination=destination,
            compile_report=compile_report,
            replay_report=replay_report,
        ),
        label="source fit",
    )
    compile_compatibility = dict(compile_report)
    compile_compatibility["status"] = {
        "T10_2_1_FRESH_INTEGRITY_COMPLETE": "T10_2_FRESH_INTEGRITY_COMPLETE",
        "PASS_T10_2_1_QA": "PASS_T10_2_QA",
    }.get(
        str(compile_compatibility.get("status")),
        compile_compatibility.get("status"),
    )
    compile_compatibility = signed_payload(
        compile_compatibility, checksum_key="report_checksum"
    )
    replay_compatibility = dict(replay_report)
    replay_compatibility["status"] = "T10_2_SOURCE_REPLAY_COMPLETE"
    replay_compatibility = signed_payload(
        replay_compatibility, checksum_key="report_checksum"
    )
    with _legacy_bindings() as legacy_runtime:
        with tempfile.TemporaryDirectory(prefix="sage-t10-2-1-train-") as temp:
            compile_path = Path(temp) / "compile_report.json"
            replay_path = Path(temp) / "replay_report.json"
            write_compact_json(compile_path, compile_compatibility)
            write_compact_json(replay_path, replay_compatibility)
            report = _t10_2.source_train_phase(
                trainer=legacy_runtime.run_source_trainer,
                manifest_path=manifest_path,
                output_dir=destination,
                compile_report_path=compile_path,
                replay_report_path=replay_path,
                repo_root=repo_root,
            )
    report = dict(report)
    report["status"] = {
        "PASS_T10_2_SOURCE_GATE": "PASS_T10_2_1_SOURCE_GATE",
        "FAIL_T10_2_SOURCE_GATE": "FAIL_T10_2_1_SOURCE_GATE",
    }.get(str(report.get("status")), report.get("status"))
    if report.get("verdict") == "PASS_T10_2_SOURCE_GATE":
        report["verdict"] = "PASS_T10_2_1_SOURCE_GATE"
    report["registered_randomness"] = _registered_randomness_spec()
    if "compile_report_checksum" in report:
        report["compile_report_checksum"] = compile_report.get("report_checksum")
    report["replay_report_checksum"] = replay_report.get("report_checksum")
    inputs = report.get("inputs")
    if isinstance(inputs, Mapping):
        normalized_inputs = dict(inputs)
        normalized_inputs["compile_report"] = _t10_2.artifact_descriptor(
            destination / "compile_report.json"
        )
        normalized_inputs["replay_report"] = _t10_2.artifact_descriptor(
            destination / "replay_report.json"
        )
        report["inputs"] = normalized_inputs
    report = signed_payload(report, checksum_key="report_checksum")
    write_compact_json(source_path, report)
    return _reconstruct_existing_source_report(
        manifest=manifest,
        destination=destination,
    )


def _require_source_gate(
    *, manifest: Mapping[str, Any], source_report_path: Path, output_dir: Path
) -> dict[str, Any]:
    report = _read_signed_json(source_report_path, checksum_key="report_checksum")
    if report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("source report/manifest binding drifted")
    checks = report.get("checks")
    controls = report.get("registered_controls")
    firewall = report.get("firewall")
    if not (
        report.get("status") == "PASS_T10_2_1_SOURCE_GATE"
        and report.get("verdict") == "PASS_T10_2_1_SOURCE_GATE"
        and report.get("passed") is True
        and isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values())
        and isinstance(controls, Mapping)
        and set(controls) == set(REGISTERED_SOURCE_CONTROLS)
        and all(value is True for value in controls.values())
        and report.get("registered_randomness") == _registered_randomness_spec()
        and isinstance(firewall, Mapping)
        and firewall.get("source_validation_opened") is True
        and not any(
            bool(firewall.get(key))
            for key in ("ar25_opened", "holdout_opened", "production_authority")
        )
    ):
        raise GateRefusalError("validation refused: T10.2.1 source gate did not pass")
    destination = Path(output_dir)
    inputs = report.get("inputs")
    metrics = report.get("metrics")
    if not isinstance(inputs, Mapping) or not isinstance(metrics, Mapping):
        raise ManifestDriftError("source gate lacks reconstructable evidence")
    evidence_paths = {
        "compile_report": destination / "compile_report.json",
        "replay_report": destination / "replay_report.json",
        "fresh_events": destination / "source_events.jsonl",
        "replay_events": destination / "replay_events.jsonl",
        "cross_fit_audit": destination / CROSS_FIT_AUDIT_FILENAME,
    }
    for key, artifact_path in evidence_paths.items():
        if inputs.get(key) != _t10_2.artifact_descriptor(artifact_path):
            raise ManifestDriftError(f"source gate artifact drifted: {key}")
    compile_report = _read_signed_json(
        evidence_paths["compile_report"], checksum_key="report_checksum"
    )
    replay_report = _read_signed_json(
        evidence_paths["replay_report"], checksum_key="report_checksum"
    )
    _verify_source_fit_opening_marker(
        path=destination / SOURCE_FIT_OPENING_MARKER_FILENAME,
        manifest=manifest,
        destination=destination,
        compile_report=compile_report,
        replay_report=replay_report,
    )
    if metrics.get("registered_randomness") != _registered_randomness_spec():
        raise ManifestDriftError("source metrics randomness binding drifted")
    recipe_binding = metrics.get("challenger_recipe")
    if not isinstance(recipe_binding, Mapping) or recipe_binding.get("bound") is not True:
        raise ManifestDriftError("source gate lacks a bound challenger recipe")
    recipe_path = destination / CHALLENGER_RECIPE_FILENAME
    if (
        recipe_binding.get("path") != CHALLENGER_RECIPE_FILENAME
        or recipe_binding.get("artifact") != _t10_2.artifact_descriptor(recipe_path)
    ):
        raise ManifestDriftError("challenger recipe artifact drifted")
    recipe = _read_signed_json(recipe_path, checksum_key="recipe_checksum")
    if (
        recipe_binding.get("recipe_checksum") != recipe.get("recipe_checksum")
        or recipe.get("manifest_checksum") != manifest.get("manifest_checksum")
        or recipe.get("registered_randomness") != _registered_randomness_spec()
    ):
        raise ManifestDriftError("challenger recipe binding drifted")
    with _legacy_bindings() as legacy_runtime:
        _t10_2._derive_source_control_views(metrics)
        _t10_2._verify_source_evidence_binding(
            metrics,
            manifest=manifest,
            fresh_path=evidence_paths["fresh_events"],
            replay_path=evidence_paths["replay_events"],
            cross_fit_path=evidence_paths["cross_fit_audit"],
        )
        legacy_runtime._rebuild_challenger_posterior(
            recipe=recipe,
            output_dir=destination,
            manifest=manifest,
        )
        compatibility = dict(report)
        compatibility["status"] = "PASS_T10_2_SOURCE_GATE"
        compatibility["verdict"] = "PASS_T10_2_SOURCE_GATE"
        _t10_2._reconstruct_source_report(
            manifest=manifest,
            source_report=compatibility,
            destination=destination,
            limits=DEFAULT_RESOURCE_LIMITS,
        )
    return report


def _validation_opening_payload(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    return signed_payload(
        {
            "format_version": VALIDATION_OPENING_MARKER_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "source_report": _t10_2.artifact_descriptor(source_path),
            "source_report_checksum": source_report["report_checksum"],
            "validation_plan_sha256": canonical_sha256(_validation_plan()),
            "opened_once": True,
        },
        checksum_key="opening_checksum",
    )


def _create_one_shot_marker(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Claim one preregistered irreversible phase opening durably."""

    path.parent.mkdir(parents=True, exist_ok=True)
    material = (canonical_json(payload) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise GateRefusalError(
            f"{label} has already consumed its one registered opening"
        ) from exc
    try:
        offset = 0
        while offset < len(material):
            written = os.write(descriptor, material[offset:])
            if written <= 0:
                raise OSError("validation marker write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_validation_opening_marker(
    *,
    path: Path,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    marker = _read_signed_json(path, checksum_key="opening_checksum")
    expected = _validation_opening_payload(
        manifest=manifest,
        source_report=source_report,
        source_path=source_path,
    )
    if marker != expected:
        raise ManifestDriftError("validation opening marker drifted")
    return marker


def _reconstruct_existing_validation_report(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
    destination: Path,
) -> dict[str, Any]:
    validation_path = destination / "validation_report.json"
    validation = _read_signed_json(
        validation_path, checksum_key="report_checksum"
    )
    status = str(validation.get("status", ""))
    passed = validation.get("passed") is True
    verdict = str(validation.get("verdict", ""))
    expected_passed = status == "PASS_T10_2_1_VALIDATION"
    if not (
        validation.get("manifest_checksum") == manifest.get("manifest_checksum")
        and validation.get("source_report_checksum")
        == source_report.get("report_checksum")
        and status
        in {"PASS_T10_2_1_VALIDATION", "FAIL_T10_2_1_VALIDATION"}
        and passed is expected_passed
        and (
            verdict == "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED"
        )
        is expected_passed
        and (
            expected_passed
            or verdict
            in {
                "SOURCE_VALIDATION_TRANSFER_MISS",
                "SAFETY_OR_RESOURCE_MISS",
            }
        )
    ):
        raise ManifestDriftError("validation terminal status drifted")
    source_compatibility = dict(source_report)
    source_compatibility["status"] = "PASS_T10_2_SOURCE_GATE"
    source_compatibility["verdict"] = "PASS_T10_2_SOURCE_GATE"
    validation_compatibility = dict(validation)
    validation_compatibility["status"] = (
        "PASS_T10_2_VALIDATION"
        if expected_passed
        else "FAIL_T10_2_VALIDATION"
    )
    if expected_passed:
        validation_compatibility["verdict"] = (
            "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED"
        )
    with _legacy_bindings():
        _t10_2._reconstruct_validation_report(
            manifest=manifest,
            source_report=source_compatibility,
            source_path=source_path,
            validation_report=validation_compatibility,
            destination=destination,
        )
    return validation


def validate_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    validation_started = time.perf_counter()
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    source_path = Path(source_report_path or destination / "source_report.json")
    canonical_source_path = destination / "source_report.json"
    if source_path.resolve() != canonical_source_path.resolve():
        raise ManifestDriftError(
            "validation requires the canonical source report path"
        )
    source_report = _require_source_gate(
        manifest=manifest,
        source_report_path=source_path,
        output_dir=destination,
    )
    marker_path = destination / VALIDATION_OPENING_MARKER_FILENAME
    validation_path = destination / "validation_report.json"
    validation_evidence = (
        validation_path,
        destination / "validation_runs.jsonl",
        destination / "validation_timing_proof.json",
    )
    if marker_path.exists():
        _verify_validation_opening_marker(
            path=marker_path,
            manifest=manifest,
            source_report=source_report,
            source_path=source_path,
        )
        if validation_path.is_file():
            return _reconstruct_existing_validation_report(
                manifest=manifest,
                source_report=source_report,
                source_path=source_path,
                destination=destination,
            )
        raise GateRefusalError(
            "validation was opened but did not reach a reconstructible terminal report"
        )
    if any(path.exists() for path in validation_evidence):
        raise ManifestDriftError(
            "validation evidence exists without its one-shot opening marker"
        )
    _create_one_shot_marker(
        marker_path,
        _validation_opening_payload(
            manifest=manifest,
            source_report=source_report,
            source_path=source_path,
        ),
        label="validation",
    )
    source_compatibility = dict(source_report)
    source_compatibility["status"] = "PASS_T10_2_SOURCE_GATE"
    source_compatibility["verdict"] = "PASS_T10_2_SOURCE_GATE"
    source_compatibility = signed_payload(
        source_compatibility, checksum_key="report_checksum"
    )
    with _legacy_bindings() as legacy_runtime:
        with tempfile.TemporaryDirectory(prefix="sage-t10-2-1-validate-") as temp:
            compatibility_path = Path(temp) / "source_report.json"
            write_compact_json(compatibility_path, source_compatibility)
            baseline = legacy_runtime.T10_1BehaviorFrozenPolicyFactory(
                repo_root=repo_root
            )
            challenger = legacy_runtime.T10_2GaugePolicyFactory(
                source_report=source_compatibility,
                manifest=manifest,
                output_dir=destination,
            )
            report = _t10_2.validate_phase(
                manifest_path=manifest_path,
                output_dir=destination,
                source_report_path=compatibility_path,
                repo_root=repo_root,
                env_factory=legacy_runtime.T10_2ValidationFactory(
                    source_report=source_compatibility,
                    manifest=manifest,
                    t10_1_policy_factory=baseline,
                    t10_2_policy_factory=challenger,
                ),
            )
    validation_finished = time.perf_counter()
    if (
        not isinstance(validation_started, float)
        or not isinstance(validation_finished, float)
        or not all(
            value >= 0.0 and value < float("inf")
            for value in (validation_started, validation_finished)
        )
        or validation_finished < validation_started
    ):
        raise DataGateError("external validation monotonic clock regressed")
    timing_path = destination / "validation_timing_proof.json"
    timing = _read_signed_json(timing_path, checksum_key="timing_proof_checksum")
    timing["source_report"] = _t10_2.artifact_descriptor(source_path)
    timing["source_report_checksum"] = source_report["report_checksum"]
    timing_code = dict(timing.get("code_sha256", {}))
    for code_path in (
        "theory/sage_t/t10_2_1_protocol.py",
        "theory/sage_t/t10_2_1_runtime.py",
    ):
        timing_code[code_path] = manifest["portable_code_sha256"][code_path]
    timing["code_sha256"] = timing_code
    timing["monotonic_started"] = validation_started
    timing["monotonic_finished"] = validation_finished
    timing["monotonic_elapsed_seconds"] = (
        validation_finished - validation_started
    )
    timing = signed_payload(timing, checksum_key="timing_proof_checksum")
    write_compact_json(timing_path, timing)
    inherited_report = dict(report)
    metrics = inherited_report.get("metrics")
    inputs = inherited_report.get("inputs")
    if not isinstance(metrics, Mapping) or not isinstance(inputs, Mapping):
        raise ManifestDriftError(
            "validation result lacks reconstructible metrics or inputs"
        )
    adjusted_metrics = dict(metrics)
    adjusted_metrics["wall_seconds"] = timing["monotonic_elapsed_seconds"]
    reconstruction_source = dict(source_report)
    reconstruction_source["status"] = "PASS_T10_2_SOURCE_GATE"
    reconstruction_source["verdict"] = "PASS_T10_2_SOURCE_GATE"
    with _legacy_bindings():
        report = _t10_2.build_validation_report(
            manifest=manifest,
            source_report=reconstruction_source,
            metrics=adjusted_metrics,
        )
    report["inputs"] = dict(inputs)
    report = signed_payload(report, checksum_key="report_checksum")
    report = dict(report)
    passed = report.get("status") == "PASS_T10_2_VALIDATION"
    report["status"] = (
        "PASS_T10_2_1_VALIDATION" if passed else "FAIL_T10_2_1_VALIDATION"
    )
    if passed:
        report["verdict"] = "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED"
    report["source_report_checksum"] = source_report["report_checksum"]
    inputs = report.get("inputs")
    if isinstance(inputs, Mapping):
        normalized_inputs = dict(inputs)
        normalized_inputs["source_report"] = _t10_2.artifact_descriptor(source_path)
        normalized_inputs["validation_timing_proof"] = (
            _t10_2.artifact_descriptor(timing_path)
        )
        report["inputs"] = normalized_inputs
    report = signed_payload(report, checksum_key="report_checksum")
    write_compact_json(validation_path, report)
    return _reconstruct_existing_validation_report(
        manifest=manifest,
        source_report=source_report,
        source_path=source_path,
        destination=destination,
    )


def _reconstruct_negative_source_report(
    *,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    destination: Path,
) -> None:
    if source.get("terminal_stage") == "acquisition":
        compile_path = destination / "compile_report.json"
        compile_report = _read_signed_json(
            compile_path, checksum_key="report_checksum"
        )
        collection_path = destination / "collection_report.json"
        collection = _read_signed_json(
            collection_path, checksum_key="report_checksum"
        )
        expected_compile = _build_acquisition_failure_report(
            manifest=manifest,
            collection=collection,
            output_dir=destination,
        )
        expected_source = _build_terminal_source_failure_report(
            manifest=manifest,
            compile_report=expected_compile,
            output_dir=destination,
        )
        forbidden_downstream = (
            destination / "replay_report.json",
            destination / "replay_events.jsonl",
            destination / CHALLENGER_RECIPE_FILENAME,
            destination / SOURCE_FIT_OPENING_MARKER_FILENAME,
            destination / VALIDATION_OPENING_MARKER_FILENAME,
            destination / "validation_report.json",
            destination / "validation_runs.jsonl",
            destination / "validation_timing_proof.json",
        )
        if (
            collection.get("manifest_checksum")
            != manifest.get("manifest_checksum")
            or canonical_json(compile_report) != canonical_json(expected_compile)
            or canonical_json(source) != canonical_json(expected_source)
            or any(path.exists() for path in forbidden_downstream)
        ):
            raise ManifestDriftError(
                "terminal acquisition source report did not reconstruct"
            )
        return

    compile_report = _read_signed_json(
        destination / "compile_report.json", checksum_key="report_checksum"
    )
    replay_report = _read_signed_json(
        destination / "replay_report.json", checksum_key="report_checksum"
    )
    _verify_source_fit_opening_marker(
        path=destination / SOURCE_FIT_OPENING_MARKER_FILENAME,
        manifest=manifest,
        destination=destination,
        compile_report=compile_report,
        replay_report=replay_report,
    )
    compatibility = dict(source)
    if compatibility.get("status") == "FAIL_T10_2_1_SOURCE_GATE":
        compatibility["status"] = "FAIL_T10_2_SOURCE_GATE"
    if compatibility.get("registered_randomness") != _registered_randomness_spec():
        raise ManifestDriftError("negative source randomness binding drifted")
    with _legacy_bindings():
        _t10_2._reconstruct_source_report(
            manifest=manifest,
            source_report=compatibility,
            destination=destination,
            limits=DEFAULT_RESOURCE_LIMITS,
        )


def _archive_publication_marker(destination: Path) -> None:
    binding_path = destination.parent / REPORT_INVENTORY_BINDING_FILENAME
    if not binding_path.is_file():
        return
    digest = raw_file_sha256(binding_path)
    archive_dir = destination / "publication_markers"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"superseded_{digest}.json"
    if archived.exists():
        if archived.read_bytes() != binding_path.read_bytes():
            raise ManifestDriftError("publication marker archive collision")
        index = 1
        while (archive_dir / f"superseded_{digest}_{index}.json").exists():
            index += 1
        archived = archive_dir / f"superseded_{digest}_{index}.json"
    os.replace(binding_path, archived)


def verify_publication_binding(
    *,
    manifest: Mapping[str, Any],
    destination: Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    from .t10_2_1_artifact_inventory import (
        FORMAT_VERSION as INVENTORY_FORMAT_VERSION,
        REPORT_INVENTORY_BINDING_FORMAT_VERSION,
        build_inventory,
        canonical_json as inventory_canonical_json,
        is_registered_publishable_compact_json,
    )

    root = Path(repo_root or _repo_root()).resolve()
    binding_path = destination.parent / REPORT_INVENTORY_BINDING_FILENAME
    inventory_path = destination.parent / ARTIFACT_INVENTORY_FILENAME
    report_path = destination / "report.json"
    binding = _read_signed_json(binding_path, checksum_key="binding_checksum")
    inventory = _read_signed_json(
        inventory_path, checksum_key="inventory_checksum"
    )
    report = _read_signed_json(report_path, checksum_key="report_checksum")
    rebuilt_inventory = build_inventory(
        repository_root=root,
        artifact_root=destination,
        output_path=inventory_path,
    )
    if not (
        set(binding)
        == {
            "format_version",
            "manifest_checksum",
            "report",
            "inventory",
            "inventory_checksum",
            "cycle_free",
            "binding_checksum",
        }
        and binding.get("format_version")
        == REPORT_INVENTORY_BINDING_FORMAT_VERSION
        and binding.get("manifest_checksum")
        == manifest.get("manifest_checksum")
        and binding.get("report") == _t10_2.artifact_descriptor(report_path)
        and binding.get("inventory") == _t10_2.artifact_descriptor(inventory_path)
        and binding.get("inventory_checksum") == inventory.get("inventory_checksum")
        and binding.get("cycle_free") is True
        and inventory.get("format_version") == INVENTORY_FORMAT_VERSION
        and inventory_canonical_json(inventory)
        == inventory_canonical_json(rebuilt_inventory)
        and is_registered_publishable_compact_json(report_path)
        and is_registered_publishable_compact_json(inventory_path)
        and is_registered_publishable_compact_json(binding_path)
        and report.get("status") == "T10_2_1_COMPLETE"
        and report.get("manifest_checksum") == manifest.get("manifest_checksum")
    ):
        raise ManifestDriftError("final publication binding did not reconstruct")
    return binding


def report_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_report_path: str | Path | None = None,
    validation_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    from .t10_2_1_artifact_inventory import (
        PUBLISHABLE_REPORT_NAMES,
        REPORT_INVENTORY_BINDING_FORMAT_VERSION,
        REPORT_INVENTORY_BINDING_NAME,
        build_inventory,
        write_inventory,
    )

    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    source_path = Path(source_report_path or destination / "source_report.json")
    canonical_source_path = destination / "source_report.json"
    if source_path.resolve() != canonical_source_path.resolve():
        raise ManifestDriftError(
            "final report requires the canonical source report path"
        )
    source = _read_signed_json(source_path, checksum_key="report_checksum")
    if source.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("source report/manifest binding drifted")
    validation: dict[str, Any] | None = None
    validation_path = Path(
        validation_report_path or destination / "validation_report.json"
    )
    canonical_validation_path = destination / "validation_report.json"
    if validation_path.resolve() != canonical_validation_path.resolve():
        raise ManifestDriftError(
            "final report requires the canonical validation report path"
        )
    source_passed = bool(
        source.get("status") == "PASS_T10_2_1_SOURCE_GATE"
        and source.get("verdict") == "PASS_T10_2_1_SOURCE_GATE"
        and source.get("passed") is True
    )
    if source_passed:
        source = _require_source_gate(
            manifest=manifest,
            source_report_path=source_path,
            output_dir=destination,
        )
        _verify_validation_opening_marker(
            path=destination / VALIDATION_OPENING_MARKER_FILENAME,
            manifest=manifest,
            source_report=source,
            source_path=source_path,
        )
        validation = _read_signed_json(
            validation_path, checksum_key="report_checksum"
        )
        if validation.get("manifest_checksum") != manifest["manifest_checksum"]:
            raise ManifestDriftError("validation/manifest binding drifted")
        if validation.get("source_report_checksum") != source.get(
            "report_checksum"
        ):
            raise ManifestDriftError("validation/source report binding drifted")
        verdict = str(validation.get("verdict", "SOURCE_VALIDATION_TRANSFER_MISS"))
        if validation.get("status") not in {
            "PASS_T10_2_1_VALIDATION",
            "FAIL_T10_2_1_VALIDATION",
        } or verdict not in {
            "SOURCE_VALIDATION_TRANSFER_MISS",
            "SAFETY_OR_RESOURCE_MISS",
            "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED",
        }:
            raise ManifestDriftError("validation verdict ladder drifted")
        validation_checks = validation.get("checks")
        validation_firewall = validation.get("firewall")
        supported = bool(
            validation.get("status") == "PASS_T10_2_1_VALIDATION"
            and validation.get("passed") is True
            and verdict == "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED"
            and isinstance(validation_checks, Mapping)
            and validation_checks
            and all(value is True for value in validation_checks.values())
            and isinstance(validation_firewall, Mapping)
            and validation_firewall.get("source_validation_opened") is True
            and not any(
                bool(validation_firewall.get(key))
                for key in (
                    "ar25_opened",
                    "holdout_opened",
                    "production_authority",
                )
            )
        )
        validation_status_passed = (
            validation.get("status") == "PASS_T10_2_1_VALIDATION"
        )
        validation_verdict_supported = (
            verdict == "SAGE_T10_2_1_GAUGE_POSTERIOR_SUPPORTED"
        )
        if not (
            validation.get("passed") is validation_status_passed
            and validation_verdict_supported is validation_status_passed
        ):
            raise ManifestDriftError(
                "validation status/pass/verdict equivalence drifted"
            )
        source_compatibility = dict(source)
        source_compatibility["status"] = "PASS_T10_2_SOURCE_GATE"
        source_compatibility["verdict"] = "PASS_T10_2_SOURCE_GATE"
        validation_compatibility = dict(validation)
        validation_compatibility["status"] = (
            "PASS_T10_2_VALIDATION"
            if validation_status_passed
            else "FAIL_T10_2_VALIDATION"
        )
        if validation_verdict_supported:
            validation_compatibility["verdict"] = (
                "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED"
            )
        with _legacy_bindings():
            _t10_2._reconstruct_validation_report(
                manifest=manifest,
                source_report=source_compatibility,
                source_path=source_path,
                validation_report=validation_compatibility,
                destination=destination,
            )
    else:
        forbidden_stale_validation = (
            destination / VALIDATION_OPENING_MARKER_FILENAME,
            destination / "validation_report.json",
            destination / "validation_timing_proof.json",
            destination / "validation_runs.jsonl",
        )
        if any(path.exists() for path in forbidden_stale_validation):
            raise ManifestDriftError(
                "validation artifacts exist behind a closed source gate"
            )
        _reconstruct_negative_source_report(
            manifest=manifest,
            source=source,
            destination=destination,
        )
        verdict = str(source.get("verdict", "DATA_OR_PROVENANCE_INVALID"))
        expected_statuses = {
            "DATA_OR_PROVENANCE_INVALID": "DATA_OR_PROVENANCE_INVALID",
            "SOURCE_ACQUISITION_OR_RESOURCE_MISS": (
                "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
            ),
        }
        expected_status = expected_statuses.get(
            verdict, "FAIL_T10_2_1_SOURCE_GATE"
        )
        source_firewall = source.get("firewall")
        if (
            verdict not in SOURCE_NEGATIVE_VERDICTS
            or source.get("status") != expected_status
            or source.get("passed") is not False
            or not isinstance(source_firewall, Mapping)
            or any(
                bool(source_firewall.get(key))
                for key in (
                    "source_validation_opened",
                    "ar25_opened",
                    "holdout_opened",
                    "production_authority",
                )
            )
        ):
            raise ManifestDriftError("source verdict ladder drifted")
        supported = False
    report = signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "phase": "report",
            "status": "T10_2_1_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "source_report_checksum": source["report_checksum"],
            "validation_report_checksum": (
                None if validation is None else validation["report_checksum"]
            ),
            "verdict": verdict,
            "supported": supported,
            "evidence_boundary": (
                "scientific support requires source causal gates and paired active "
                "validation; deployment authority remains closed"
            ),
            "inputs": {
                "source_report": _t10_2.artifact_descriptor(source_path),
                **(
                    {}
                    if validation is None
                    else {
                        "validation_report": _t10_2.artifact_descriptor(
                            validation_path
                        )
                    }
                ),
            },
            "artifact_inventory": {
                "inventory_sidecar": f"../{ARTIFACT_INVENTORY_FILENAME}",
                "binding_sidecar": f"../{REPORT_INVENTORY_BINDING_FILENAME}",
                "cyclic_hash_dependency": False,
            },
            "firewall": {
                "source_validation_opened": validation is not None,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )
    _archive_publication_marker(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    write_compact_json(report_path, report)
    root = Path(repo_root or _repo_root()).resolve()
    inventory_output_path = destination.parent / ARTIFACT_INVENTORY_FILENAME
    inventory = build_inventory(
        repository_root=root,
        artifact_root=destination,
        output_path=inventory_output_path,
    )
    lifecycle_checksum_keys = {
        CROSS_FIT_AUDIT_FILENAME: "audit_checksum",
        "validation_timing_proof.json": "timing_proof_checksum",
    }
    for lifecycle_name in PUBLISHABLE_REPORT_NAMES:
        lifecycle_path = destination / lifecycle_name
        if not lifecycle_path.is_file():
            continue
        lifecycle_payload = _read_signed_json(
            lifecycle_path,
            checksum_key=lifecycle_checksum_keys.get(
                lifecycle_name, "report_checksum"
            ),
        )
        if lifecycle_payload.get("manifest_checksum") != manifest["manifest_checksum"]:
            raise ManifestDriftError(
                f"lifecycle report/manifest binding drifted: {lifecycle_name}"
            )
    included_names = {
        Path(str(item.get("path", ""))).name
        for item in inventory.get("included_compact_reports", ())
        if isinstance(item, Mapping)
    }
    present_lifecycle_names = {
        name
        for name in PUBLISHABLE_REPORT_NAMES
        if (destination / name).is_file()
    }
    if not present_lifecycle_names.issubset(included_names):
        omitted_lifecycle = sorted(present_lifecycle_names - included_names)
        raise ManifestDriftError(
            "invalid lifecycle reports cannot be bound: "
            + ",".join(omitted_lifecycle)
        )
    budget_audit = inventory.get("budget_audit")
    if not isinstance(budget_audit, Mapping) or budget_audit.get("passed") is not True:
        raise ResourceGateError("artifact inventory exceeds registered budgets")
    inventory_path = write_inventory(
        inventory,
        repository_root=root,
        output_path=inventory_output_path,
    )
    binding = signed_payload(
        {
            "format_version": REPORT_INVENTORY_BINDING_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "report": _t10_2.artifact_descriptor(report_path),
            "inventory": _t10_2.artifact_descriptor(inventory_path),
            "inventory_checksum": inventory["inventory_checksum"],
            "cycle_free": True,
        },
        checksum_key="binding_checksum",
    )
    binding_path = destination.parent / REPORT_INVENTORY_BINDING_NAME
    write_compact_json(binding_path, binding)
    verify_publication_binding(
        manifest=manifest,
        destination=destination,
        repo_root=root,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-report")
    parser.add_argument("--validation-report")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    try:
        if args.phase == "freeze":
            payload = freeze_manifest(
                output_path=args.manifest,
                repo_root=args.repo_root,
                verify_repository=True,
            )
        elif args.phase == "collect":
            from .t10_2_1_runtime import T10_2_1SourceFactory

            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = collect_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
                env_factory=T10_2_1SourceFactory(manifest=manifest),
            )
        elif args.phase == "compile":
            payload = compile_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
            )
        elif args.phase == "replay":
            payload = replay_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
            )
        elif args.phase == "source-train":
            payload = source_train_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
            )
        elif args.phase == "validate":
            payload = validate_phase(
                manifest_path=args.manifest,
                output_dir=output,
                source_report_path=args.source_report,
                repo_root=args.repo_root,
            )
        else:
            payload = report_phase(
                manifest_path=args.manifest,
                output_dir=output,
                source_report_path=args.source_report,
                validation_report_path=args.validation_report,
                repo_root=args.repo_root,
            )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(
            canonical_json(
                {
                    "error": f"{type(exc).__name__}:{exc}",
                    "phase": args.phase,
                }
            )
        )
        return 2
    print(canonical_json(payload))
    return 0


def collect_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
    env_factory: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Late-bound to the durable runtime to keep ``freeze`` import-only."""

    from .t10_2_1_runtime import collect_phase as durable_collect_phase

    manifest = load_manifest(manifest_path, repo_root=repo_root)
    destination = _registered_output_dir(
        manifest=manifest, output_dir=output_dir, repo_root=repo_root
    )
    _refuse_upstream_mutation_after_validation(
        destination=destination, phase="collect"
    )
    _refuse_upstream_mutation_after_source_fit(
        destination=destination, phase="collect"
    )
    return durable_collect_phase(
        manifest_path=manifest_path,
        output_dir=destination,
        repo_root=repo_root,
        env_factory=env_factory,
        **kwargs,
    )


__all__ = [
    "ARTIFACT_INVENTORY_FILENAME",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_FORMAT_VERSION",
    "CONFIRMATION_SEEDS",
    "CROSS_FIT_AUDIT_FILENAME",
    "DEFAULT_CODE_FILES",
    "DEFAULT_INPUT_FILES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_SEEDS",
    "EXCLUSIVE_VERDICTS",
    "FIT_SEED",
    "FORMAT_VERSION",
    "HASH_ALGORITHM",
    "JOURNAL_FORMAT_VERSION",
    "LANE_REPORT_FORMAT_VERSION",
    "PERMUTATION_SEED",
    "PHASES",
    "REPORT_INVENTORY_BINDING_FILENAME",
    "RESET_REPORT_FORMAT_VERSION",
    "SOURCE_GAMES",
    "SOURCE_LANE_TIMEOUT_SECONDS",
    "SOURCE_MAXIMUM_ACTIONS",
    "SOURCE_MAXIMUM_WALL_SECONDS",
    "SOURCE_STOP_NEW_ACTIONS_SECONDS",
    "VALIDATION_GAMES",
    "VALIDATION_SEEDS",
    "build_manifest",
    "canonical_file_sha256",
    "canonical_json",
    "canonical_sha256",
    "collect_phase",
    "compile_phase",
    "enforce_environment_firewall",
    "freeze_manifest",
    "load_manifest",
    "main",
    "ordered_jsonl_root_sha256",
    "raw_file_sha256",
    "replay_phase",
    "report_phase",
    "signed_payload",
    "source_train_phase",
    "validate_phase",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
