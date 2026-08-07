"""Preregistered orchestration and evidence firewall for SAGE.T10.2.

The module deliberately separates the seven registered phases.  There is no
``all`` shortcut: every phase consumes checksummed artifacts from its
predecessors and fails closed on drift, forbidden evidence, or a failed gate.
Live environments are dependency-injected.  The CLI binds the repository's
registered local source runtime lazily, while validation still refuses unless
both exact behavior-frozen policy factories are available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

FORMAT_VERSION = "sage-t10.2-preregistered-protocol-v1"
EVENT_FORMAT_VERSION = "sage-t10.2-physical-event-v2"
COMPACT_PROJECTION_FORMAT_VERSION = "sage-t10.2-runtime-v2"
COMPACT_QUOTIENT_FORMAT_VERSION = "sage-t10.2-structural-quotient-v2"
MAXIMUM_MODEL_VIEW_BYTES = 32 * 1_024
MAXIMUM_COMPACT_EVENT_BYTES = 48 * 1_024
BASELINE_COMMIT = "05c1c91b82054af55a03ef962745f4f101cd3c0e"
BASELINE_FROZEN_SHA256 = {
    "theory/sage_t/contracts.py": (
        "d4a6957c3a7aab3c4fb798316acbe42375db42d7997275fecc10126d353dfee9"
    ),
    "theory/sage_t/posterior.py": (
        "aedb4dad969517e03d3afd0190b3ad6a509f80b1cdc43a7350c0de0c05036e01"
    ),
    "theory/sage_t/executor.py": (
        "1cbe93ecf85da169a45a8c39e9b50ca051671f33589d14e1d5394e5823b4ddf2"
    ),
    "theory/sage_t/decision.py": (
        "fda110efd864fe8036b233d18855bf4ad090bf504cde40955b957be25f5bd595"
    ),
    "theory/sage_t/controller.py": (
        "70f18cc6966ec76a489d28e6370fb5d26a417d95405ba94b039a63567921ed39"
    ),
}

PHASES = (
    "freeze",
    "collect",
    "compile",
    "replay",
    "source-train",
    "validate",
    "report",
)

SOURCE_GAMES = (
    "bp35-0a0ad940",
    "lp85-305b61c3",
    "su15-4c352900",
)
VALIDATION_GAMES = (
    "re86-4e57566e",
    "ls20-9607627b",
    "sc25-f9b21a2f",
)
AR25_GAME = "ar25-e3c63847"

DISCOVERY_SEEDS = (0, 1, 2)
CONFIRMATION_SEEDS = (3, 4, 5)
SOURCE_RESETS_PER_GAME_SEED = 4
SOURCE_ACTIONS_PER_RESET = 64
SOURCE_MAXIMUM_ACTIONS = 4_608
SOURCE_PRECOLLECTION_ABORTED_ACTIONS = 1
SOURCE_MAXIMUM_NEW_ACTIONS = (
    SOURCE_MAXIMUM_ACTIONS - SOURCE_PRECOLLECTION_ABORTED_ACTIONS
)
SOURCE_MAXIMUM_WALL_SECONDS = 5_400

VALIDATION_SEEDS = (2101, 2102, 2103, 2104, 2105)
VALIDATION_RESETS_PER_GAME_SEED = 14
VALIDATION_ACTIONS_PER_RESET = 96
VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER = 20_160
VALIDATION_MAXIMUM_WALL_SECONDS = 21_600
VALIDATION_EXEMPT_STOP_REASONS = frozenset(
    {"progression", "game_over", "terminal", "option_exhausted"}
)
VALIDATION_REGISTERED_STOP_REASONS = frozenset(
    {*VALIDATION_EXEMPT_STOP_REASONS, "budget_exhausted"}
)
VALIDATION_UNREGISTERED_STOP_REASONS = frozenset(
    {
        "policy_abstained",
        "no_legal_actions",
        "decision_error",
        "illegal_action",
        "step_error",
        "observation_error",
    }
)
VALIDATION_STOP_REASONS = frozenset(
    {*VALIDATION_REGISTERED_STOP_REASONS, *VALIDATION_UNREGISTERED_STOP_REASONS}
)

MIB = 1024**2
GIB = 1024**3

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_2_protocol_manifest.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_2_gauge_posterior"
CHALLENGER_RECIPE_FILENAME = "t10_2_challenger_recipe.json"
CROSS_FIT_AUDIT_FILENAME = "cross_fit_audit.json"
CROSS_FIT_AUDIT_FORMAT_VERSION = "sage-t10.2-cross-fit-audit-v1"
ARTIFACT_INVENTORY_FILENAME = "t10_2_omitted_artifacts_inventory.json"
REPORT_INVENTORY_BINDING_FILENAME = "t10_2_report_inventory_binding.json"
VALIDATION_TIMING_PROOF_FILENAME = "validation_timing_proof.json"
VALIDATION_TIMING_PROOF_FORMAT_VERSION = "sage-t10.2-validation-timing-proof-v1"
DEFAULT_PROTOCOL_PATH = (
    Path("reports") / "SAGE_T10_2_FULL_RELATIONAL_GAUGE_POSTERIOR_PROTOCOL.md"
)

DEFAULT_CODE_FILES = (
    "theory/sage_t/contracts.py",
    "theory/sage_t/posterior.py",
    "theory/sage_t/executor.py",
    "theory/sage_t/decision.py",
    "theory/sage_t/controller.py",
    "theory/sage_t/observer_frames_v10_2.py",
    "theory/sage_t/frame_transport_v10_2.py",
    "theory/sage_t/frame_adapters_v10_2.py",
    "theory/sage_t/mixed_automata_v10_2.py",
    "theory/sage_t/gauge_inference_v10_2.py",
    "theory/sage_t/compact_quotient_v10_2.py",
    "theory/sage_t/factorized_posterior_v10_2.py",
    "theory/sage_t/t10_2_artifact_inventory.py",
    "theory/sage_t/t10_2_protocol.py",
    "theory/sage_t/compiler.py",
    "theory/sage_t/progress_witness_v10.py",
    "theory/live_transition_loop.py",
    "theory/sage12/scene_graph.py",
    "theory/sage12/mt/graph.py",
    "theory/sage12/mt/transition.py",
    "theory/sage12/topological_invariants_v4_19.py",
    "theory/m1/polymorphic_a25_adapter.py",
    "theory/m2/m3_execution_smoke.py",
    "theory/non_ar25_active_micro_run.py",
    "theory/real_env_option_adapter.py",
    "theory/unified_cognition_ab_benchmark.py",
    "tests/test_sage_t_observer_frames_v10_2.py",
    "tests/test_sage_t_frame_transport_v10_2.py",
    "tests/test_sage_t_frame_adapters_v10_2.py",
    "tests/test_sage_t_mixed_automata_v10_2.py",
    "tests/test_sage_t_gauge_inference_v10_2.py",
    "tests/test_sage_t_compact_quotient_v10_2.py",
    "tests/test_sage_t_factorized_posterior_v10_2.py",
    "tests/test_sage_t_t10_2_artifact_inventory.py",
    "tests/test_sage_t_t10_2_protocol.py",
)
OPTIONAL_CODE_FILES = (
    "theory/sage_t/t10_2_runtime.py",
    "tests/test_sage_t_t10_2_runtime.py",
)
DEFAULT_SOURCE_SHARD_FILES = {
    "bp35-0a0ad940": (
        "training/sage12/bound_mechanic_pilot_v4_3/source_train_shards/bp35.jsonl"
    ),
    "lp85-305b61c3": (
        "training/sage12/bound_mechanic_pilot_v4_3/source_train_shards/lp85.jsonl"
    ),
    "su15-4c352900": (
        "training/sage12/bound_mechanic_pilot_v4_3/source_train_shards/su15.jsonl"
    ),
}
DEFAULT_SOURCE_METADATA_FILES = {
    "bp35-0a0ad940": "environment_files/bp35/0a0ad940/metadata.json",
    "lp85-305b61c3": "environment_files/lp85/305b61c3/metadata.json",
    "su15-4c352900": "environment_files/su15/4c352900/metadata.json",
}
DEFAULT_INPUT_FILES = (
    ".gitattributes",
    "reports/SAGE_T10_1_BASELINE.md",
    "reports/SAGE_T10_2_FULL_RELATIONAL_GAUGE_POSTERIOR_PROTOCOL.md",
    *DEFAULT_SOURCE_SHARD_FILES.values(),
)

REGISTERED_SOURCE_CONTROLS = (
    "t10_1_behavior_frozen_baseline",
    "capacity_matched_independent_posterior",
    "single_frame_root_only",
    "single_frame_allocentric_object_relative",
    "single_frame_action_aligned_relational",
    "single_frame_action_rooted_topological",
    "identity_only_transport",
    "no_transport",
    "deterministically_permuted_transport",
    "frame_swap",
    "binding_swap",
    "dynamics_swap",
    "goal_swap",
    "option_swap",
    "early_map_collapse",
    "immediate_noop_deduplication",
    "best_executed_sequence_oracle",
    "grammar_oracle",
    "transport_oracle",
    "dynamics_oracle",
    "goal_oracle",
    "option_oracle",
    "complete_program_oracle",
)

RARE_EVALUATION_ANCHORS = frozenset({"level_complete", "win", "game_over"})
ALLOWED_SOURCE_KINDS = frozenset({"fresh_source_trajectory", "frozen_source_replay"})
REPLAY_SPLIT = "frozen_source_replay_v4_3"
REGISTERED_FRAME_ORDER = (
    "root_only",
    "allocentric_object_relative",
    "action_aligned_relational",
    "action_rooted_topological",
)
REGISTERED_FRAME_IDS = frozenset(REGISTERED_FRAME_ORDER)
REPLAY_CONVERSION_CODE_PATHS = {
    "converter": "theory/sage_t/t10_2_runtime.py",
    "projector": "theory/sage_t/frame_adapters_v10_2.py",
    "observer_frames": "theory/sage_t/observer_frames_v10_2.py",
    "compiler_contract": "theory/sage_t/contracts.py",
    "compiler": "theory/sage_t/compiler.py",
}
VALIDATION_TIMING_CODE_PATHS = (
    "theory/sage_t/t10_2_protocol.py",
    "theory/sage_t/t10_2_runtime.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GAME_ID_LITERAL = re.compile(
    r"(?<![a-z0-9])[a-z]{2}\d{2}(?:-[0-9a-f]{8})?(?![a-z0-9])",
    re.IGNORECASE,
)
_COORDINATE_LITERAL = re.compile(
    r"(?<![a-z0-9])\(?\s*-?\d+\s*,\s*-?\d+\s*\)?(?![a-z0-9])",
    re.IGNORECASE,
)
_UUID_LITERAL = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])",
    re.IGNORECASE,
)


class ProtocolError(RuntimeError):
    """Base class for a registered-protocol refusal."""


class ManifestDriftError(ProtocolError):
    """Raised when frozen code, inputs, environment, or manifest drift."""


class FirewallError(ProtocolError):
    """Raised before forbidden evidence or an environment can be opened."""


class ResourceGateError(ProtocolError):
    """Raised before a registered resource bound can be exceeded."""


class DataGateError(ProtocolError):
    """Raised for malformed, duplicated, contaminated, or unsigned evidence."""


class GateRefusalError(ProtocolError):
    """Raised when a later phase lacks a passing predecessor gate."""


class RuntimeUnavailableError(ProtocolError):
    """Raised when no explicit live-runtime adapter was injected."""


_MODEL_VIEW_FIELDS = frozenset({"frames"})
_MODEL_FRAME_FIELDS = frozenset(
    {
        "before",
        "after",
        "before_hash",
        "after_hash",
        "observation",
        "observation_hash",
        "complete",
        "missing",
        "covered_channels",
        "provenance",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "format_version",
        "frame_id",
        "before_hash",
        "after_hash",
        "observation",
        "observation_hash",
        "complete",
        "missing",
        "covered_channels",
        "provenance",
        "canonical_hash",
    }
)
_STRUCTURAL_OBSERVATION_FIELDS = frozenset(
    {
        "object_deltas",
        "relation_deltas",
        "topology_deltas",
        "known_channels",
        "residual",
    }
)
_STRUCTURAL_OBSERVATION_CHANNELS = frozenset({"objects", "relations", "topology"})
_CLOSED_FRAME_CHANNELS = frozenset(
    {
        "entities",
        "facts",
        "counters",
        "registers",
        "topology",
        "regime",
        "objects",
        "relations",
    }
)
_TRANSPORT_CERTIFICATE_FIELDS = frozenset(
    {
        "source_frame",
        "target_frame",
        "transport_hash",
        "certificate_hash",
        "coverage",
        "exact",
        "comparable",
        "mapping_kind",
        "round_trip_exact",
        "certifies_gauge_equivalence",
        "projection_complete",
        "live_graph_exact_attested",
        "summary_commutative_exact",
        "commutativity",
    }
)
_COMMUTATIVITY_CERTIFICATE_FIELDS = frozenset({"before", "after", "dynamics", "exact"})
_TRANSPORT_SUMMARY_FIELDS = frozenset(
    {
        "mapping_kind",
        "comparable",
        "round_trip_exact",
        "entity_permutation_invariant",
        "commutative_exact",
        "live_graph_exact_attested",
        "summary_commutative_exact",
        "certificate_count",
        "exact_certificate_count",
        "partial_certificate_count",
        "identity_root_certificate_exact",
    }
)


@dataclass(frozen=True)
class ResourceLimits:
    """Frozen T10.2 storage, memory, and free-space limits."""

    maximum_ledger_bytes: int = 256 * MIB
    maximum_shard_bytes: int = 64 * MIB
    maximum_checkpoint_bytes: int = 10 * MIB
    maximum_derived_file_bytes: int = 512 * MIB
    maximum_scratch_bytes: int = 5 * GIB
    maximum_cache_bytes: int = 5 * GIB
    maximum_repository_bytes: int = 12 * GIB
    maximum_resident_bytes: int = 8 * GIB
    minimum_free_bytes: int = 100 * GIB


DEFAULT_RESOURCE_LIMITS = ResourceLimits()


@dataclass(frozen=True)
class ResourceSnapshot:
    repository_bytes: int
    scratch_bytes: int
    cache_bytes: int
    resident_bytes: int
    free_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    """Return the only JSON encoding admitted for T10.2 artifacts."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_file_sha256(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return canonical_sha256(payload)


def signed_payload(payload: Mapping[str, Any], *, checksum_key: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(checksum_key, None)
    result[checksum_key] = canonical_sha256(result)
    return result


def write_compact_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_lines(destination, (canonical_json(payload) + "\n",))


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    """Atomically replace a text artifact after flushing its complete temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def read_checked_json(
    path: str | Path,
    *,
    checksum_key: str = "report_checksum",
    require_canonical: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestDriftError(f"invalid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ManifestDriftError(f"expected JSON object: {source}")
    if require_canonical and raw != canonical_json(payload) + "\n":
        raise ManifestDriftError(f"non-canonical JSON: {source}")
    unsigned = dict(payload)
    checksum = str(unsigned.pop(checksum_key, ""))
    if not checksum or checksum != canonical_sha256(unsigned):
        raise ManifestDriftError(f"checksum mismatch: {source}")
    return payload


def artifact_descriptor(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {"bytes": source.stat().st_size, "sha256": file_sha256(source)}


def _verified_challenger_recipe_binding(
    metrics: Mapping[str, Any],
    *,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    limits: ResourceLimits,
) -> dict[str, Any]:
    """Bind trainer output to one compact immutable validation recipe.

    A trainer-provided ``bound`` flag is never trusted.  The protocol resolves
    the one registered filename itself, verifies both checksums and the
    manifest binding, and only then records authorization evidence.
    """

    raw = metrics.get("challenger_recipe")
    supplied = dict(raw) if isinstance(raw, Mapping) else {}
    result: dict[str, Any] = {
        "bound": False,
        "path": CHALLENGER_RECIPE_FILENAME,
        "artifact": supplied.get("artifact"),
        "recipe_checksum": supplied.get("recipe_checksum"),
    }
    try:
        if supplied.get("path") != CHALLENGER_RECIPE_FILENAME:
            raise ManifestDriftError("challenger recipe path is not registered")
        recipe_path = Path(output_dir) / CHALLENGER_RECIPE_FILENAME
        enforce_artifact_limit(recipe_path, kind="checkpoint", limits=limits)
        descriptor = artifact_descriptor(recipe_path)
        if supplied.get("artifact") != descriptor:
            raise ManifestDriftError("challenger recipe artifact drifted")
        recipe = read_checked_json(
            recipe_path,
            checksum_key="recipe_checksum",
        )
        if recipe.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ManifestDriftError("challenger recipe/manifest binding drifted")
        if recipe.get("kind") != "immutable_source_posterior_recipe":
            raise ManifestDriftError("unsupported challenger recipe kind")
        if supplied.get("recipe_checksum") != recipe.get("recipe_checksum"):
            raise ManifestDriftError("challenger recipe checksum drifted")
    except (OSError, ProtocolError, ValueError, KeyError) as exc:
        result["binding_error"] = type(exc).__name__
        return result
    return {
        "bound": True,
        "path": CHALLENGER_RECIPE_FILENAME,
        "artifact": descriptor,
        "recipe_checksum": recipe["recipe_checksum"],
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )


def _relative_name(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _hash_named_paths(
    root: Path,
    paths: Sequence[str | Path],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths:
        path = _resolve(root, item)
        if not path.is_file():
            raise ManifestDriftError(f"registered input missing: {path}")
        hashes[_relative_name(root, path)] = file_sha256(path)
    return dict(sorted(hashes.items()))


def code_hashes(
    repo_root: str | Path | None = None,
    *,
    paths: Sequence[str | Path] = DEFAULT_CODE_FILES,
) -> dict[str, str]:
    root = Path(repo_root or _repo_root()).resolve()
    selected = list(paths)
    if tuple(paths) == DEFAULT_CODE_FILES:
        selected.extend(
            item for item in OPTIONAL_CODE_FILES if _resolve(root, item).is_file()
        )
    return _hash_named_paths(root, selected)


def input_hashes(
    repo_root: str | Path | None = None,
    *,
    paths: Sequence[str | Path] = DEFAULT_INPUT_FILES,
) -> dict[str, str]:
    return _hash_named_paths(Path(repo_root or _repo_root()).resolve(), paths)


def environment_metadata() -> dict[str, Any]:
    """Return stable runtime metadata; volatile paths and timestamps are absent."""

    runtime_versions: dict[str, str] = {}
    for distribution in ("arc-agi", "arcengine"):
        try:
            runtime_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[distribution] = "unavailable"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "byteorder": sys.byteorder,
        "hash_algorithm": "sha256",
        "json_encoding": "canonical-compact-ascii-v1",
        "runtime_versions": runtime_versions,
    }


def _verify_baseline_repository(root: Path) -> None:
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{BASELINE_COMMIT}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestDriftError(
            f"baseline {BASELINE_COMMIT} is unavailable or not an ancestor"
        ) from exc
    if resolved != BASELINE_COMMIT:
        raise ManifestDriftError("baseline commit resolved to an unexpected object")


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
        _verify_baseline_repository(root)
    env = dict(environment or environment_metadata())
    current_code_hashes = code_hashes(root, paths=code_paths)
    current_frozen_hashes = _hash_named_paths(root, tuple(BASELINE_FROZEN_SHA256))
    for path, expected in BASELINE_FROZEN_SHA256.items():
        if current_frozen_hashes.get(path) != expected:
            raise ManifestDriftError(f"frozen baseline file drifted: {path}")
    source_shards = {
        game: {
            "path": path,
            "sha256": file_sha256(_resolve(root, path)),
        }
        for game, path in DEFAULT_SOURCE_SHARD_FILES.items()
    }
    source_metadata = {
        game: {
            "path": path,
            "canonical_json_sha256": canonical_json_file_sha256(_resolve(root, path)),
        }
        for game, path in DEFAULT_SOURCE_METADATA_FILES.items()
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T10_2_COLLECTION",
        "baseline_commit": BASELINE_COMMIT,
        "baseline_frozen_code_sha256": dict(BASELINE_FROZEN_SHA256),
        "registered_phases": list(PHASES),
        "code_sha256": current_code_hashes,
        "input_sha256": input_hashes(root, paths=input_paths),
        "environment": env,
        "environment_sha256": canonical_sha256(env),
        "frozen_source_shards": source_shards,
        "source_environment_metadata": source_metadata,
        "artifact_contract": {
            "physical_event_format": EVENT_FORMAT_VERSION,
            "projection_format": COMPACT_PROJECTION_FORMAT_VERSION,
            "structural_quotient_format": COMPACT_QUOTIENT_FORMAT_VERSION,
            "observer_frames": list(REGISTERED_FRAME_ORDER),
            "maximum_model_view_bytes": MAXIMUM_MODEL_VIEW_BYTES,
            "maximum_compact_event_bytes": MAXIMUM_COMPACT_EVENT_BYTES,
            "raw_frames_persisted": False,
            "full_graphs_persisted": False,
        },
        "source_plan": {
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
            "maximum_actions": SOURCE_MAXIMUM_ACTIONS,
            "precollection_aborted_actions": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
            "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
            "maximum_wall_seconds": SOURCE_MAXIMUM_WALL_SECONDS,
        },
        "validation_plan": {
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
        },
        "qa_gate": {
            "minimum_confident_correspondence": 0.90,
            "maximum_fully_ambiguous_correspondence": 0.10,
            "minimum_predicate_prevalence": 0.005,
            "maximum_predicate_prevalence": 0.95,
            "minimum_predicate_support": 32,
            "minimum_predicate_games": 2,
            "minimum_evaluable_nonterminal_prefix_fraction": 0.80,
            "minimum_multiframe_coherent_prefix_fraction": 0.50,
        },
        "source_gate": {
            "minimum_grammar_progress_games": 2,
            "minimum_grammar_levels": 2,
            "maximum_positive_fold_rank": 8,
            "maximum_median_positive_fold_rank": 4,
            "minimum_oracle_level_recovery": 0.50,
            "minimum_nonnegative_games": 2,
            "maximum_game_seed_probe_accuracy_increment": 0.10,
            "registered_controls": list(REGISTERED_SOURCE_CONTROLS),
        },
        "validation_gate": {
            "minimum_total_level_advantage": 1,
            "minimum_nonnegative_games": 2,
            "minimum_completed_budget_fraction": 0.95,
            "maximum_decision_p95_ms": 750.0,
            "maximum_decision_p99_ms": 2_500.0,
            "maximum_observation_p95_ms": 500.0,
            "maximum_observation_p99_ms": 3_000.0,
            "maximum_wall_seconds": VALIDATION_MAXIMUM_WALL_SECONDS,
        },
        "resource_limits": asdict(DEFAULT_RESOURCE_LIMITS),
        "firewall": {
            "source_train_games": list(SOURCE_GAMES),
            "source_validation_games": list(VALIDATION_GAMES),
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
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
    manifest = build_manifest(
        repo_root=repo_root,
        code_paths=code_paths,
        input_paths=input_paths,
        environment=environment,
        verify_repository=verify_repository,
    )
    write_compact_json(output_path, manifest)
    enforce_artifact_limit(output_path, kind="derived")
    return manifest


def _assert_exact_manifest_tree(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    """Reject missing, extra, mistyped, or changed preregistered constants."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ManifestDriftError(f"T10.2 {path} must be a mapping")
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise ManifestDriftError(f"T10.2 {path} key registry drifted")
        for key, expected_value in expected.items():
            _assert_exact_manifest_tree(
                actual[key],
                expected_value,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ManifestDriftError(f"T10.2 {path} sequence drifted")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact_manifest_tree(
                actual_value,
                expected_value,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ManifestDriftError(f"T10.2 {path} constant drifted")


def _validate_manifest_constants(manifest: Mapping[str, Any]) -> None:
    for key, expected in (
        ("format_version", FORMAT_VERSION),
        ("status", "FROZEN_BEFORE_T10_2_COLLECTION"),
        ("baseline_commit", BASELINE_COMMIT),
        ("baseline_frozen_code_sha256", BASELINE_FROZEN_SHA256),
        ("registered_phases", list(PHASES)),
    ):
        _assert_exact_manifest_tree(manifest.get(key), expected, path=key)

    expected_source_plan = {
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
        "maximum_actions": SOURCE_MAXIMUM_ACTIONS,
        "precollection_aborted_actions": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
        "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
        "maximum_wall_seconds": SOURCE_MAXIMUM_WALL_SECONDS,
    }
    expected_validation_plan = {
        "games": list(VALIDATION_GAMES),
        "seeds": list(VALIDATION_SEEDS),
        "resets_per_game_seed": VALIDATION_RESETS_PER_GAME_SEED,
        "maximum_actions_per_reset": VALIDATION_ACTIONS_PER_RESET,
        "maximum_actions_per_controller": VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER,
        "maximum_wall_seconds": VALIDATION_MAXIMUM_WALL_SECONDS,
        "counterbalanced": True,
        "posterior_reset_between_pairs": True,
        "learning_between_seeds_or_controllers": False,
    }
    expected_qa_gate = {
        "minimum_confident_correspondence": 0.90,
        "maximum_fully_ambiguous_correspondence": 0.10,
        "minimum_predicate_prevalence": 0.005,
        "maximum_predicate_prevalence": 0.95,
        "minimum_predicate_support": 32,
        "minimum_predicate_games": 2,
        "minimum_evaluable_nonterminal_prefix_fraction": 0.80,
        "minimum_multiframe_coherent_prefix_fraction": 0.50,
    }
    expected_source_gate = {
        "minimum_grammar_progress_games": 2,
        "minimum_grammar_levels": 2,
        "maximum_positive_fold_rank": 8,
        "maximum_median_positive_fold_rank": 4,
        "minimum_oracle_level_recovery": 0.50,
        "minimum_nonnegative_games": 2,
        "maximum_game_seed_probe_accuracy_increment": 0.10,
        "registered_controls": list(REGISTERED_SOURCE_CONTROLS),
    }
    expected_validation_gate = {
        "minimum_total_level_advantage": 1,
        "minimum_nonnegative_games": 2,
        "minimum_completed_budget_fraction": 0.95,
        "maximum_decision_p95_ms": 750.0,
        "maximum_decision_p99_ms": 2_500.0,
        "maximum_observation_p95_ms": 500.0,
        "maximum_observation_p99_ms": 3_000.0,
        "maximum_wall_seconds": VALIDATION_MAXIMUM_WALL_SECONDS,
    }
    expected_firewall = {
        "source_train_games": list(SOURCE_GAMES),
        "source_validation_games": list(VALIDATION_GAMES),
        "source_validation_opened": False,
        "ar25_opened": False,
        "holdout_opened": False,
        "production_authority": False,
    }
    expected_artifact_contract = {
        "physical_event_format": EVENT_FORMAT_VERSION,
        "projection_format": COMPACT_PROJECTION_FORMAT_VERSION,
        "structural_quotient_format": COMPACT_QUOTIENT_FORMAT_VERSION,
        "observer_frames": list(REGISTERED_FRAME_ORDER),
        "maximum_model_view_bytes": MAXIMUM_MODEL_VIEW_BYTES,
        "maximum_compact_event_bytes": MAXIMUM_COMPACT_EVENT_BYTES,
        "raw_frames_persisted": False,
        "full_graphs_persisted": False,
    }
    for key, expected in (
        ("artifact_contract", expected_artifact_contract),
        ("source_plan", expected_source_plan),
        ("validation_plan", expected_validation_plan),
        ("qa_gate", expected_qa_gate),
        ("source_gate", expected_source_gate),
        ("validation_gate", expected_validation_gate),
        ("resource_limits", asdict(DEFAULT_RESOURCE_LIMITS)),
        ("firewall", expected_firewall),
    ):
        _assert_exact_manifest_tree(manifest.get(key), expected, path=key)

    shards = manifest.get("frozen_source_shards")
    metadata = manifest.get("source_environment_metadata")
    if not isinstance(shards, Mapping) or set(shards) != set(SOURCE_GAMES):
        raise ManifestDriftError("T10.2 frozen source shard registry drifted")
    if not isinstance(metadata, Mapping) or set(metadata) != set(SOURCE_GAMES):
        raise ManifestDriftError("T10.2 source metadata registry drifted")
    for game in SOURCE_GAMES:
        shard = shards[game]
        source_metadata = metadata[game]
        if not isinstance(shard, Mapping) or set(shard) != {"path", "sha256"}:
            raise ManifestDriftError(
                f"T10.2 frozen source shard schema drifted: {game}"
            )
        if shard.get("path") != DEFAULT_SOURCE_SHARD_FILES[
            game
        ] or not _SHA256.fullmatch(str(shard.get("sha256", ""))):
            raise ManifestDriftError(
                f"T10.2 frozen source shard binding drifted: {game}"
            )
        if not isinstance(source_metadata, Mapping) or set(source_metadata) != {
            "path",
            "canonical_json_sha256",
        }:
            raise ManifestDriftError(f"T10.2 source metadata schema drifted: {game}")
        if source_metadata.get("path") != DEFAULT_SOURCE_METADATA_FILES[
            game
        ] or not _SHA256.fullmatch(
            str(source_metadata.get("canonical_json_sha256", ""))
        ):
            raise ManifestDriftError(f"T10.2 source metadata binding drifted: {game}")


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
    manifest = read_checked_json(path, checksum_key="manifest_checksum")
    _validate_manifest_constants(manifest)
    root = Path(repo_root or _repo_root()).resolve()
    if verify_repository:
        _verify_baseline_repository(root)
    if verify_code and manifest.get("code_sha256") != code_hashes(
        root, paths=code_paths
    ):
        raise ManifestDriftError("T10.2 code hash drifted")
    if verify_inputs and manifest.get("input_sha256") != input_hashes(
        root, paths=input_paths
    ):
        raise ManifestDriftError("T10.2 registered input hash drifted")
    if verify_inputs:
        expected_shards = {
            game: {
                "path": registered_path,
                "sha256": file_sha256(_resolve(root, registered_path)),
            }
            for game, registered_path in DEFAULT_SOURCE_SHARD_FILES.items()
        }
        if manifest.get("frozen_source_shards") != expected_shards:
            raise ManifestDriftError("T10.2 frozen source shard binding drifted")
        expected_metadata = {
            game: {
                "path": registered_path,
                "canonical_json_sha256": canonical_json_file_sha256(
                    _resolve(root, registered_path)
                ),
            }
            for game, registered_path in DEFAULT_SOURCE_METADATA_FILES.items()
        }
        if manifest.get("source_environment_metadata") != expected_metadata:
            raise ManifestDriftError("T10.2 source metadata binding drifted")
    if verify_environment:
        current = dict(environment or environment_metadata())
        if manifest.get("environment") != current or manifest.get(
            "environment_sha256"
        ) != canonical_sha256(current):
            raise ManifestDriftError("T10.2 environment hash drifted")
    return manifest


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file() and not candidate.is_symlink():
            total += candidate.stat().st_size
    return total


def _resident_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return 0


def resource_snapshot(repo_root: str | Path | None = None) -> ResourceSnapshot:
    root = Path(repo_root or _repo_root()).resolve()
    usage = shutil.disk_usage(root)
    return ResourceSnapshot(
        repository_bytes=_directory_size(root),
        scratch_bytes=_directory_size(root / ".sage_t_scratch" / "t10_2"),
        cache_bytes=_directory_size(root / ".sage_t_cache" / "t10_2"),
        resident_bytes=_resident_bytes(),
        free_bytes=int(usage.free),
    )


def enforce_resource_limits(
    snapshot: ResourceSnapshot,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    expensive: bool,
) -> None:
    checks = {
        "repository": (snapshot.repository_bytes, limits.maximum_repository_bytes),
        "scratch": (snapshot.scratch_bytes, limits.maximum_scratch_bytes),
        "cache": (snapshot.cache_bytes, limits.maximum_cache_bytes),
        "resident": (snapshot.resident_bytes, limits.maximum_resident_bytes),
    }
    for name, (observed, maximum) in checks.items():
        if observed > maximum:
            raise ResourceGateError(f"{name} budget exceeded: {observed} > {maximum}")
    if expensive and snapshot.free_bytes < limits.minimum_free_bytes:
        raise ResourceGateError(
            f"free-disk floor crossed: {snapshot.free_bytes} < "
            f"{limits.minimum_free_bytes}"
        )


def guard_resources(
    *,
    repo_root: str | Path | None = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    expensive: bool,
    probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
) -> ResourceSnapshot:
    snapshot = probe(repo_root)
    enforce_resource_limits(snapshot, limits=limits, expensive=expensive)
    return snapshot


def enforce_artifact_limit(
    path: str | Path,
    *,
    kind: str,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    maxima = {
        "ledger": limits.maximum_ledger_bytes,
        "shard": limits.maximum_shard_bytes,
        "checkpoint": limits.maximum_checkpoint_bytes,
        "derived": limits.maximum_derived_file_bytes,
    }
    if kind not in maxima:
        raise ValueError(f"unknown artifact kind: {kind}")
    size = Path(path).stat().st_size
    if size > maxima[kind]:
        raise ResourceGateError(
            f"{kind} artifact exceeds limit: {size} > {maxima[kind]}"
        )


def enforce_output_artifacts(
    output_dir: str | Path,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    directory = Path(output_dir)
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.name.casefold()
        if path.suffix.casefold() in {".pt", ".pth", ".ckpt"} or any(
            token in name for token in ("checkpoint", "posterior")
        ):
            kind = "checkpoint"
        elif "shard" in name:
            kind = "shard"
        elif path.suffix.casefold() == ".jsonl":
            kind = "ledger"
        else:
            kind = "derived"
        enforce_artifact_limit(path, kind=kind, limits=limits)


def enforce_environment_firewall(
    *, phase: str, game_id: str, source_gate_passed: bool = False
) -> None:
    """Authorize a game before an environment factory is invoked."""

    game = str(game_id)
    if game == AR25_GAME or game.startswith("ar25-"):
        raise FirewallError("ar25 is permanently closed to T10.2")
    if "holdout" in game.casefold():
        raise FirewallError("the final holdout is permanently closed to T10.2")
    if phase == "collect":
        if game not in SOURCE_GAMES:
            raise FirewallError(f"non-source game blocked before collection: {game}")
        return
    if phase == "validate":
        if not source_gate_passed:
            raise GateRefusalError("source gate has not authorized validation")
        if game not in VALIDATION_GAMES:
            raise FirewallError(f"unregistered validation game blocked: {game}")
        return
    raise FirewallError(f"environment access is not registered for phase {phase!r}")


def _walk_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk_strings(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def forbidden_transfer_payload_hits(payload: Any, *, root: str) -> tuple[str, ...]:
    """Recursively reject raw encodings and instance identities."""

    forbidden_keys = {
        "absolute_coordinate",
        "absolute_coordinates",
        "absolute_position",
        "absolute_x",
        "absolute_y",
        "adjacency_matrix",
        "cells",
        "color",
        "colors",
        "colour",
        "colours",
        "edges",
        "entities",
        "entity_ids",
        "entity_id",
        "frame_pixels",
        "full_graph",
        "game",
        "game_id",
        "graph",
        "graphs",
        "grid",
        "identities",
        "node_id",
        "nodes",
        "object_id",
        "object_ids",
        "palette",
        "persistent_entity_id",
        "persistent_id",
        "persistent_ids",
        "pixel",
        "pixels",
        "raw_frame",
        "raw_frames",
        "raw_graph",
        "raw_grid",
        "true_facts",
        "false_facts",
        "seed",
        "seed_id",
        "uuid",
        "x",
        "y",
    }
    hits: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).casefold().replace("-", "_")
                child = f"{path}.{raw_key}" if path else str(raw_key)
                if key in forbidden_keys:
                    hits.append(child)
                visit(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            folded = value.casefold()
            if (
                value in SOURCE_GAMES
                or value in VALIDATION_GAMES
                or value == AR25_GAME
                or _GAME_ID_LITERAL.search(value)
                or _COORDINATE_LITERAL.search(value)
                or _UUID_LITERAL.search(value)
                or any(
                    token in folded
                    for token in (
                        "game_id",
                        "seed_id",
                        "entity_id",
                        "object_id",
                        "persistent_id",
                    )
                )
            ):
                hits.append(path)

    visit(payload, root)
    return tuple(sorted(set(hits)))


def forbidden_model_view_hits(model_view: Any) -> tuple[str, ...]:
    """Backward-compatible name for the recursive transfer firewall."""

    return forbidden_transfer_payload_hits(model_view, root="model_view")


def _require_sha256(value: Any, *, label: str, event_id: str) -> str:
    digest = str(value).casefold()
    if not _SHA256.fullmatch(digest):
        raise DataGateError(f"invalid {label} SHA-256: {event_id}")
    return digest


def _require_nonnegative_int(value: Any, *, label: str, event_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataGateError(f"invalid {label}: {event_id}")
    result = value
    if result < 0:
        raise DataGateError(f"invalid {label}: {event_id}")
    return result


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
    event_id: str,
) -> None:
    observed = set(value)
    if observed != set(required):
        raise DataGateError(
            f"{label} schema drifted for {event_id}; "
            f"missing={sorted(required - observed)}, "
            f"unknown={sorted(observed - required, key=str)}"
        )


def _require_canonical_string_list(
    value: Any,
    *,
    label: str,
    event_id: str,
    allowed: frozenset[str] | None = None,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DataGateError(f"invalid {label}: {event_id}")
    if not allow_empty and not value:
        raise DataGateError(f"empty {label}: {event_id}")
    if value != sorted(set(value)):
        raise DataGateError(f"non-canonical {label}: {event_id}")
    if allowed is not None and not set(value) <= set(allowed):
        raise DataGateError(f"unknown {label}: {event_id}")
    return list(value)


def _validate_structural_observation(
    value: Any,
    *,
    event_id: str,
    frame_id: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataGateError(
            f"projection {frame_id} lacks structural observation: {event_id}"
        )
    _require_exact_fields(
        value,
        required=_STRUCTURAL_OBSERVATION_FIELDS,
        label=f"projection {frame_id} observation",
        event_id=event_id,
    )
    for field in ("object_deltas", "relation_deltas", "topology_deltas"):
        rows = value[field]
        if not isinstance(rows, Mapping) or any(
            not isinstance(key, str) or not key.strip() for key in rows
        ):
            raise DataGateError(f"invalid projection {frame_id} {field}: {event_id}")
        if list(rows) != sorted(rows):
            raise DataGateError(
                f"non-canonical projection {frame_id} {field}: {event_id}"
            )
        for amount in rows.values():
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
            ):
                raise DataGateError(
                    f"non-finite projection {frame_id} {field}: {event_id}"
                )
    _require_canonical_string_list(
        value["known_channels"],
        label=f"projection {frame_id} observation channels",
        event_id=event_id,
        allowed=_STRUCTURAL_OBSERVATION_CHANNELS,
    )
    residual = value["residual"]
    if not isinstance(residual, list) or any(
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
        for amount in residual
    ):
        raise DataGateError(f"invalid projection {frame_id} residual: {event_id}")
    return value


def _validate_compact_model_view(
    model_view: Any,
    *,
    event_id: str,
) -> Mapping[str, Mapping[str, Any]]:
    from .compact_quotient_v10_2 import assert_compact_quotient, quotient_sha256

    if not isinstance(model_view, Mapping):
        raise DataGateError(f"event lacks a compact model_view: {event_id}")
    hits = forbidden_model_view_hits(model_view)
    if hits:
        raise FirewallError(
            f"identity-bearing model_view for {event_id}: {', '.join(hits)}"
        )
    _require_exact_fields(
        model_view,
        required=_MODEL_VIEW_FIELDS,
        label="model_view",
        event_id=event_id,
    )
    try:
        model_bytes = len(canonical_json(model_view).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise DataGateError(f"model_view is not finite JSON: {event_id}") from exc
    if model_bytes > MAXIMUM_MODEL_VIEW_BYTES:
        raise DataGateError(
            f"model_view exceeds {MAXIMUM_MODEL_VIEW_BYTES} bytes: {event_id}"
        )
    frames = model_view.get("frames")
    if not isinstance(frames, Mapping) or set(map(str, frames)) != set(
        REGISTERED_FRAME_IDS
    ):
        raise DataGateError(
            f"model_view requires exactly four registered frames: {event_id}"
        )
    validated: dict[str, Mapping[str, Any]] = {}
    for frame_id in REGISTERED_FRAME_ORDER:
        frame = frames.get(frame_id)
        if not isinstance(frame, Mapping):
            raise DataGateError(f"invalid model frame {frame_id}: {event_id}")
        _require_exact_fields(
            frame,
            required=_MODEL_FRAME_FIELDS,
            label=f"model frame {frame_id}",
            event_id=event_id,
        )
        before = frame["before"]
        after = frame["after"]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise DataGateError(
                f"model frame {frame_id} lacks quotient pair: {event_id}"
            )
        try:
            assert_compact_quotient(before)
            assert_compact_quotient(after)
            expected_before = quotient_sha256(before)
            expected_after = quotient_sha256(after)
        except (TypeError, ValueError) as exc:
            raise DataGateError(
                f"invalid model frame {frame_id} quotient: {event_id}"
            ) from exc
        before_hash = _require_sha256(
            frame["before_hash"],
            label=f"model frame {frame_id} before",
            event_id=event_id,
        )
        after_hash = _require_sha256(
            frame["after_hash"],
            label=f"model frame {frame_id} after",
            event_id=event_id,
        )
        if before_hash != expected_before or after_hash != expected_after:
            raise DataGateError(
                f"model frame {frame_id} quotient hash mismatch: {event_id}"
            )
        observation = _validate_structural_observation(
            frame["observation"], event_id=event_id, frame_id=frame_id
        )
        observation_hash = _require_sha256(
            frame["observation_hash"],
            label=f"model frame {frame_id} observation",
            event_id=event_id,
        )
        if observation_hash != canonical_sha256(observation):
            raise DataGateError(
                f"model frame {frame_id} observation hash mismatch: {event_id}"
            )
        if frame["complete"] not in (True, False):
            raise DataGateError(
                f"model frame {frame_id} lacks completeness: {event_id}"
            )
        missing = _require_canonical_string_list(
            frame["missing"],
            label=f"model frame {frame_id} missing fields",
            event_id=event_id,
        )
        if frame["complete"] is True and missing:
            raise DataGateError(
                f"complete model frame {frame_id} declares missing data: {event_id}"
            )
        if frame["complete"] is False and not missing:
            raise DataGateError(
                f"partial model frame {frame_id} hides missing data: {event_id}"
            )
        _require_canonical_string_list(
            frame["covered_channels"],
            label=f"model frame {frame_id} covered channels",
            event_id=event_id,
            allowed=_CLOSED_FRAME_CHANNELS,
        )
        _require_canonical_string_list(
            frame["provenance"],
            label=f"model frame {frame_id} provenance",
            event_id=event_id,
            allow_empty=False,
        )
        validated[frame_id] = frame
    return validated


def _validate_persisted_transport_orbits(
    row: Mapping[str, Any],
    *,
    frames: Mapping[str, Mapping[str, Any]],
    event_id: str,
) -> None:
    from .frame_transport_v10_2 import TransportOrbitWitness

    envelopes = row.get("transport_orbits")
    if not isinstance(envelopes, list):
        raise DataGateError(f"transport_orbits must be a list: {event_id}")
    orbit_hashes: list[str] = []
    for envelope in envelopes:
        if not isinstance(envelope, Mapping):
            raise DataGateError(f"invalid transport orbit: {event_id}")
        try:
            witness = TransportOrbitWitness.from_persisted_attestation(envelope)
        except (TypeError, ValueError) as exc:
            raise DataGateError(
                f"invalid persisted transport attestation: {event_id}"
            ) from exc
        source = frames.get(witness.source_frame_id)
        target = frames.get(witness.target_frame_id)
        if source is None or target is None:
            raise DataGateError(f"persisted transport escaped frame bank: {event_id}")
        attestation = envelope.get("attestation")
        if not isinstance(attestation, Mapping):
            raise DataGateError(f"invalid transport attestation: {event_id}")
        bindings = {
            "source_before_summary_hash": source["before_hash"],
            "source_after_summary_hash": source["after_hash"],
            "target_before_summary_hash": target["before_hash"],
            "target_after_summary_hash": target["after_hash"],
            "source_observation_hash": source["observation_hash"],
            "target_observation_hash": target["observation_hash"],
        }
        if any(attestation.get(key) != value for key, value in bindings.items()):
            raise DataGateError(
                f"persisted transport summary binding drifted: {event_id}"
            )
        orbit_hashes.append(
            _require_sha256(
                envelope.get("orbit_hash"),
                label="transport orbit",
                event_id=event_id,
            )
        )
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        registered = provenance.get("transport_orbit_hashes")
        if registered is not None and registered != orbit_hashes:
            raise DataGateError(
                f"transport orbit provenance binding drifted: {event_id}"
            )


def _validate_transport_evidence(
    row: Mapping[str, Any],
    *,
    event_id: str,
) -> list[str]:
    certificates = row.get("transport_certificates")
    if not isinstance(certificates, list):
        raise DataGateError(f"transport_certificates must be a list: {event_id}")
    certificate_hashes: list[str] = []
    exact_count = 0
    for certificate in certificates:
        if not isinstance(certificate, Mapping):
            raise DataGateError(f"invalid transport certificate: {event_id}")
        _require_exact_fields(
            certificate,
            required=_TRANSPORT_CERTIFICATE_FIELDS,
            label="transport certificate",
            event_id=event_id,
        )
        if (
            certificate.get("source_frame") not in REGISTERED_FRAME_IDS
            or certificate.get("target_frame") not in REGISTERED_FRAME_IDS
        ):
            raise DataGateError(f"transport certificate escaped frame bank: {event_id}")
        _require_sha256(
            certificate.get("transport_hash"),
            label="transport map",
            event_id=event_id,
        )
        certificate_hashes.append(
            _require_sha256(
                certificate.get("certificate_hash"),
                label="transport certificate",
                event_id=event_id,
            )
        )
        coverage = certificate.get("coverage")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(float(coverage))
            or not 0.0 <= float(coverage) <= 1.0
        ):
            raise DataGateError(f"invalid transport coverage: {event_id}")
        boolean_fields = (
            "exact",
            "comparable",
            "round_trip_exact",
            "certifies_gauge_equivalence",
            "projection_complete",
            "live_graph_exact_attested",
            "summary_commutative_exact",
        )
        if any(certificate.get(field) not in (True, False) for field in boolean_fields):
            raise DataGateError(f"invalid transport certificate flags: {event_id}")
        commutativity = certificate.get("commutativity")
        if not isinstance(commutativity, Mapping):
            raise DataGateError(f"invalid commutativity certificate: {event_id}")
        _require_exact_fields(
            commutativity,
            required=_COMMUTATIVITY_CERTIFICATE_FIELDS,
            label="commutativity certificate",
            event_id=event_id,
        )
        for stage in ("before", "after", "dynamics"):
            _require_sha256(
                commutativity.get(stage),
                label=f"commutativity {stage}",
                event_id=event_id,
            )
        if commutativity.get("exact") not in (True, False):
            raise DataGateError(f"invalid commutativity result: {event_id}")
        exact = bool(certificate["exact"])
        expected_exact = bool(
            certificate["live_graph_exact_attested"]
            and certificate["summary_commutative_exact"]
        )
        if (
            exact is not expected_exact
            or certificate["comparable"] is not exact
            or certificate["certifies_gauge_equivalence"] is not exact
            or certificate["mapping_kind"] != ("exact" if exact else "partial")
            or (exact and certificate["round_trip_exact"] is not True)
            or (exact and commutativity["exact"] is not True)
        ):
            raise DataGateError(
                f"inconsistent transport certificate semantics: {event_id}"
            )
        exact_count += int(exact)
    if len(certificate_hashes) != len(set(certificate_hashes)):
        raise DataGateError(f"duplicate transport certificate: {event_id}")

    summary = row.get("transport")
    if not isinstance(summary, Mapping):
        raise DataGateError(f"event lacks transport summary: {event_id}")
    _require_exact_fields(
        summary,
        required=_TRANSPORT_SUMMARY_FIELDS,
        label="transport summary",
        event_id=event_id,
    )
    for field in (
        "comparable",
        "round_trip_exact",
        "entity_permutation_invariant",
        "commutative_exact",
        "live_graph_exact_attested",
        "summary_commutative_exact",
        "identity_root_certificate_exact",
    ):
        if summary.get(field) not in (True, False):
            raise DataGateError(f"invalid transport summary flag {field}: {event_id}")
    expected_total = len(certificates)
    if (
        _require_nonnegative_int(
            summary.get("certificate_count"),
            label="transport certificate_count",
            event_id=event_id,
        )
        != expected_total
        or _require_nonnegative_int(
            summary.get("exact_certificate_count"),
            label="transport exact_certificate_count",
            event_id=event_id,
        )
        != exact_count
        or _require_nonnegative_int(
            summary.get("partial_certificate_count"),
            label="transport partial_certificate_count",
            event_id=event_id,
        )
        != expected_total - exact_count
    ):
        raise DataGateError(f"transport certificate counts drifted: {event_id}")
    all_exact = bool(certificates) and exact_count == expected_total
    if (
        summary.get("mapping_kind") != ("exact" if all_exact else "partial")
        or summary.get("comparable") is not all_exact
        or summary.get("round_trip_exact") is not all_exact
        or summary.get("commutative_exact") is not all_exact
    ):
        raise DataGateError(f"transport summary semantics drifted: {event_id}")
    return certificate_hashes


def _validate_runtime_provenance_bindings(
    row: Mapping[str, Any],
    *,
    frames: Mapping[str, Mapping[str, Any]],
    event_id: str,
) -> None:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise DataGateError(f"event lacks runtime provenance: {event_id}")
    if provenance.get("collector") != COMPACT_PROJECTION_FORMAT_VERSION:
        raise DataGateError(f"collector version drifted: {event_id}")
    if provenance.get("projector_bank") != list(REGISTERED_FRAME_ORDER):
        raise DataGateError(f"projector bank drifted: {event_id}")
    expected_summaries = [
        frames[frame_id][stage]
        for frame_id in sorted(frames)
        for stage in ("before_hash", "after_hash")
    ]
    if provenance.get("summary_hashes") != expected_summaries:
        raise DataGateError(f"summary provenance binding drifted: {event_id}")
    expected_observations = [
        frames[frame_id]["observation_hash"] for frame_id in sorted(frames)
    ]
    if provenance.get("observation_hashes") != expected_observations:
        raise DataGateError(f"observation provenance binding drifted: {event_id}")
    certificate_hashes = _validate_transport_evidence(row, event_id=event_id)
    if provenance.get("transport_certificate_hashes") != certificate_hashes:
        raise DataGateError(
            f"transport certificate provenance binding drifted: {event_id}"
        )
    _require_canonical_string_list(
        provenance.get("physical_outcome_known_channels"),
        label="physical outcome known channels",
        event_id=event_id,
        allowed=frozenset({"progress", "terminal", "goal"}),
    )
    if provenance.get("raw_runtime_state_retained") is not False:
        raise FirewallError(f"raw runtime state retained: {event_id}")


def _enforce_transferable_event_firewall(
    row: Mapping[str, Any], *, event_id: str
) -> None:
    transferable_payload = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "event_checksum",
            "event_id",
            "format_version",
            "game_id",
            "provenance",
            "reset_index",
            "seed",
            "split",
            "step_index",
        }
    }
    provenance_payload = {
        key: value
        for key, value in _mapping(row.get("provenance")).items()
        if key not in {"game_id", "seed"}
    }
    transferable_payload["provenance_audit"] = provenance_payload
    hits = forbidden_transfer_payload_hits(
        transferable_payload, root="transferable_event"
    )
    if hits:
        raise FirewallError(
            f"raw or identity-bearing transferable payload for {event_id}: "
            f"{', '.join(sorted(set(hits)))}"
        )


def _validate_source_event_schema(
    row: Mapping[str, Any], *, event_id: str, fresh: bool
) -> None:
    try:
        event_bytes = len(canonical_json(row).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise DataGateError(f"event is not finite canonical JSON: {event_id}") from exc
    if event_bytes > MAXIMUM_COMPACT_EVENT_BYTES:
        raise DataGateError(
            f"event exceeds {MAXIMUM_COMPACT_EVENT_BYTES} bytes: {event_id}"
        )
    _enforce_transferable_event_firewall(row, event_id=event_id)
    reset_index = _require_nonnegative_int(
        row.get("reset_index"), label="reset_index", event_id=event_id
    )
    step_index = _require_nonnegative_int(
        row.get("step_index"), label="step_index", event_id=event_id
    )
    if fresh and reset_index >= SOURCE_RESETS_PER_GAME_SEED:
        raise DataGateError(f"source reset_index exceeds frozen budget: {event_id}")
    if fresh and step_index >= SOURCE_ACTIONS_PER_RESET:
        raise DataGateError(f"source step_index exceeds frozen budget: {event_id}")

    action = row.get("action")
    if not isinstance(action, Mapping):
        raise DataGateError(f"event lacks strict action payload: {event_id}")
    if action.get("executed") is not True:
        raise DataGateError(f"event action was not executed: {event_id}")
    for key in ("schema", "name"):
        value = action.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise DataGateError(f"event action lacks {key}: {event_id}")
    action_data = action.get("data")
    if not isinstance(action_data, Mapping):
        raise DataGateError(f"event action data is not a mapping: {event_id}")

    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping) or any(
        key not in outcome for key in ("progression", "terminal", "goal")
    ):
        raise DataGateError(f"event lacks common progression/terminal/goal: {event_id}")
    progression = outcome.get("progression")
    if (
        isinstance(progression, bool)
        or not isinstance(progression, (int, float))
        or not math.isfinite(float(progression))
    ):
        raise DataGateError(f"invalid finite progression outcome: {event_id}")
    for field in ("terminal", "goal"):
        if not isinstance(outcome.get(field), bool):
            raise DataGateError(f"invalid boolean {field} outcome: {event_id}")

    model_frames = _validate_compact_model_view(
        row.get("model_view"), event_id=event_id
    )
    projections = row.get("projections")
    if not isinstance(projections, Mapping) or set(map(str, projections)) != set(
        REGISTERED_FRAME_IDS
    ):
        raise DataGateError(
            f"event requires exactly four registered projections: {event_id}"
        )
    complete_projection_count = 0
    for frame_id in REGISTERED_FRAME_ORDER:
        projection = projections.get(frame_id)
        if not isinstance(projection, Mapping):
            raise DataGateError(f"invalid projection {frame_id}: {event_id}")
        _require_exact_fields(
            projection,
            required=_PROJECTION_FIELDS,
            label=f"projection {frame_id}",
            event_id=event_id,
        )
        if projection.get("format_version") != COMPACT_PROJECTION_FORMAT_VERSION:
            raise DataGateError(f"projection format drifted for {frame_id}: {event_id}")
        if projection.get("frame_id") != frame_id:
            raise DataGateError(
                f"projection frame binding drifted for {frame_id}: {event_id}"
            )
        canonical_hash = _require_sha256(
            projection.get("canonical_hash"),
            label=f"projection {frame_id}",
            event_id=event_id,
        )
        safe_payload = {
            key: value for key, value in projection.items() if key != "canonical_hash"
        }
        if canonical_hash != canonical_sha256(safe_payload):
            raise DataGateError(f"projection canonical hash mismatch: {event_id}")
        model_frame = model_frames[frame_id]
        shared_fields = _PROJECTION_FIELDS - {
            "format_version",
            "frame_id",
            "canonical_hash",
        }
        if any(projection[field] != model_frame[field] for field in shared_fields):
            raise DataGateError(
                f"projection/model frame binding drifted for {frame_id}: {event_id}"
            )
        _require_sha256(
            projection["before_hash"],
            label=f"projection {frame_id} before",
            event_id=event_id,
        )
        _require_sha256(
            projection["after_hash"],
            label=f"projection {frame_id} after",
            event_id=event_id,
        )
        _require_sha256(
            projection["observation_hash"],
            label=f"projection {frame_id} observation",
            event_id=event_id,
        )
        _validate_structural_observation(
            projection["observation"], event_id=event_id, frame_id=frame_id
        )
        if projection.get("complete") not in (True, False):
            raise DataGateError(f"projection lacks completeness: {event_id}")
        complete_projection_count += int(projection.get("complete") is True)
        missing = _require_canonical_string_list(
            projection["missing"],
            label=f"projection {frame_id} missing fields",
            event_id=event_id,
        )
        if projection["complete"] is True and missing:
            raise DataGateError(
                f"complete projection {frame_id} declares missing data: {event_id}"
            )
        if projection["complete"] is False and not missing:
            raise DataGateError(
                f"partial projection {frame_id} hides missing data: {event_id}"
            )
        _require_canonical_string_list(
            projection["covered_channels"],
            label=f"projection {frame_id} covered channels",
            event_id=event_id,
            allowed=_CLOSED_FRAME_CHANNELS,
        )
        _require_canonical_string_list(
            projection["provenance"],
            label=f"projection {frame_id} provenance",
            event_id=event_id,
            allow_empty=False,
        )

    _validate_persisted_transport_orbits(
        row,
        frames=model_frames,
        event_id=event_id,
    )
    _validate_runtime_provenance_bindings(
        row,
        frames=model_frames,
        event_id=event_id,
    )

    prefix = row.get("prefix")
    if not isinstance(prefix, Mapping):
        raise DataGateError(f"event lacks prefix evidence: {event_id}")
    for field in ("nonterminal", "evaluable"):
        if prefix.get(field) not in (True, False):
            raise DataGateError(f"event prefix lacks {field}: {event_id}")
    coherent_frames = _require_nonnegative_int(
        prefix.get("coherent_frames"),
        label="coherent_frames",
        event_id=event_id,
    )
    if coherent_frames != complete_projection_count:
        raise DataGateError(f"projection coherence count mismatch: {event_id}")


def _validate_replay_provenance(
    provenance: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    event_id: str,
) -> None:
    if provenance.get("source_format") != "sage12-bound-trajectory-v4.3":
        raise DataGateError(f"invalid replay source format: {event_id}")
    if provenance.get("split") != REPLAY_SPLIT:
        raise DataGateError(f"invalid replay split: {event_id}")
    descriptor = _mapping(manifest.get("frozen_source_shards", {}).get(game_id))
    registered_shard = _require_sha256(
        descriptor.get("sha256"), label="registered replay shard", event_id=event_id
    )
    observed_shard = _require_sha256(
        provenance.get("source_shard_sha256"),
        label="replay source shard",
        event_id=event_id,
    )
    if observed_shard != registered_shard:
        raise ManifestDriftError(f"replay source shard binding drifted: {event_id}")
    code_registry = _mapping(manifest.get("code_sha256"))
    expected_conversion: dict[str, str] = {}
    for role, relative_path in REPLAY_CONVERSION_CODE_PATHS.items():
        if role == "compiler" and relative_path not in code_registry:
            continue
        expected_conversion[role] = _require_sha256(
            code_registry.get(relative_path),
            label=f"manifest replay {role} code",
            event_id=event_id,
        )
    observed_conversion = provenance.get("conversion_code_sha256")
    if not isinstance(observed_conversion, Mapping) or set(
        map(str, observed_conversion)
    ) != set(expected_conversion):
        raise DataGateError(f"incomplete replay conversion code binding: {event_id}")
    for role, expected_digest in expected_conversion.items():
        observed_digest = _require_sha256(
            observed_conversion.get(role),
            label=f"replay {role} code",
            event_id=event_id,
        )
        flat_digest = _require_sha256(
            provenance.get(f"{role}_sha256"),
            label=f"flat replay {role} code",
            event_id=event_id,
        )
        if observed_digest != expected_digest or flat_digest != expected_digest:
            raise ManifestDriftError(
                f"replay conversion code binding drifted for {role}: {event_id}"
            )
    for field in (
        "source_row_sha256",
        "pair_digest",
        "trace_digest",
        "expected_pre_state_sha256",
        "replay_pre_state_sha256",
        "post_state_sha256",
        "frame_before_sha256",
        "frame_after_sha256",
    ):
        _require_sha256(provenance.get(field), label=field, event_id=event_id)
    source_line = _require_nonnegative_int(
        provenance.get("source_line"), label="source_line", event_id=event_id
    )
    if source_line < 1:
        raise DataGateError(f"invalid replay source_line: {event_id}")
    if provenance.get("arm") not in {"left", "right"}:
        raise DataGateError(f"invalid replay arm provenance: {event_id}")
    if provenance.get("raw_frames_retained") is not False:
        raise DataGateError(f"replay retained raw frames: {event_id}")
    if provenance.get("graphs_retained") is not False:
        raise DataGateError(f"replay retained full graphs: {event_id}")


def _event_id(row: Mapping[str, Any]) -> str:
    event_id = str(row.get("event_id", "")).strip()
    if not event_id:
        raise DataGateError("physical event is missing event_id")
    if len(event_id) > 256:
        raise DataGateError("physical event_id is too long")
    return event_id


def seal_event(row: Mapping[str, Any]) -> dict[str, Any]:
    event = dict(row)
    event.setdefault("format_version", EVENT_FORMAT_VERSION)
    try:
        sealed = signed_payload(event, checksum_key="event_checksum")
        encoded_bytes = len(canonical_json(sealed).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise DataGateError("physical event is not finite canonical JSON") from exc
    if encoded_bytes > MAXIMUM_COMPACT_EVENT_BYTES:
        raise DataGateError(
            f"sealed physical event exceeds {MAXIMUM_COMPACT_EVENT_BYTES} bytes"
        )
    return sealed


def verify_event_checksum(row: Mapping[str, Any]) -> None:
    unsigned = dict(row)
    checksum = str(unsigned.pop("event_checksum", ""))
    if not checksum or checksum != canonical_sha256(unsigned):
        raise DataGateError(f"event checksum mismatch: {_event_id(row)}")


def write_event_ledger(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)

    def canonical_lines() -> Iterable[str]:
        for row in rows:
            try:
                rendered = canonical_json(dict(row))
            except (TypeError, ValueError) as exc:
                raise DataGateError("event ledger contains non-finite JSON") from exc
            if len(rendered.encode("utf-8")) > MAXIMUM_COMPACT_EVENT_BYTES:
                raise DataGateError(
                    f"event ledger row exceeds {MAXIMUM_COMPACT_EVENT_BYTES} bytes"
                )
            yield rendered + "\n"

    _atomic_write_lines(
        destination,
        canonical_lines(),
    )


def read_event_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw:
            continue
        if len(raw.encode("utf-8")) > MAXIMUM_COMPACT_EVENT_BYTES:
            raise DataGateError(f"oversized JSONL row {line_number}: {path}")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataGateError(f"invalid JSONL row {line_number}: {path}") from exc
        if not isinstance(row, dict):
            raise DataGateError(f"non-object JSONL row {line_number}: {path}")
        if raw != canonical_json(row):
            raise DataGateError(f"non-canonical JSONL row {line_number}: {path}")
        rows.append(row)
    return rows


def _expected_source_split(seed: int) -> str | None:
    if seed in DISCOVERY_SEEDS:
        return "discovery"
    if seed in CONFIRMATION_SEEDS:
        return "leave_one_game_out_confirmation"
    return None


def _validate_correspondence_evidence(
    row: Mapping[str, Any],
    *,
    event_id: str,
) -> None:
    correspondence = row.get("correspondence")
    if not isinstance(correspondence, Mapping):
        raise DataGateError(f"missing correspondence evidence: {event_id}")
    denominator = _require_nonnegative_int(
        correspondence.get("fraction_denominator"),
        label="correspondence fraction_denominator",
        event_id=event_id,
    )
    confident = _require_nonnegative_int(
        correspondence.get("confident_matches"),
        label="correspondence confident_matches",
        event_id=event_id,
    )
    ambiguous = _require_nonnegative_int(
        correspondence.get("fully_ambiguous_matches"),
        label="correspondence fully_ambiguous_matches",
        event_id=event_id,
    )
    if denominator < 1:
        raise DataGateError(f"zero correspondence denominator: {event_id}")
    if confident > denominator or ambiguous > denominator:
        raise DataGateError(f"correspondence numerator exceeds denominator: {event_id}")
    if confident + ambiguous > denominator:
        raise DataGateError(f"correspondence classes overlap: {event_id}")
    for numerator, field in (
        (confident, "confident_fraction"),
        (ambiguous, "fully_ambiguous_fraction"),
    ):
        raw_fraction = correspondence.get(field)
        if isinstance(raw_fraction, bool):
            raise DataGateError(f"invalid correspondence {field}: {event_id}")
        try:
            fraction = float(raw_fraction)
        except (TypeError, ValueError) as exc:
            raise DataGateError(f"invalid correspondence {field}: {event_id}") from exc
        expected = numerator / denominator
        if (
            not math.isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
            or not math.isclose(fraction, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise DataGateError(
                f"correspondence {field} is not its registered ratio: {event_id}"
            )


def validate_source_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    replay: bool | None = None,
) -> None:
    seen: set[str] = set()
    seen_replay_sources: set[tuple[str, int, str]] = set()
    forbidden_ids = set(VALIDATION_GAMES) | {AR25_GAME}
    expected_manifest = str(manifest["manifest_checksum"])
    expected_environment = str(manifest["environment_sha256"])
    for row in rows:
        event_id = _event_id(row)
        if event_id in seen:
            raise DataGateError(f"duplicate physical event_id: {event_id}")
        seen.add(event_id)
        verify_event_checksum(row)
        if row.get("format_version") != EVENT_FORMAT_VERSION:
            raise DataGateError(f"unsupported event format: {event_id}")
        game = str(row.get("game_id", ""))
        enforce_environment_firewall(phase="collect", game_id=game)
        for path, value in _walk_strings(row.get("provenance", {})):
            if value in forbidden_ids or "holdout" in value.casefold():
                raise FirewallError(f"forbidden evidence at {path}: {value}")
        seed = _require_nonnegative_int(
            row.get("seed"), label="source seed", event_id=event_id
        )
        split = str(row.get("split", ""))
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise DataGateError(f"missing complete provenance: {event_id}")
        kind = str(provenance.get("kind", ""))
        if kind not in ALLOWED_SOURCE_KINDS:
            raise DataGateError(f"invalid source provenance kind: {event_id}")
        if replay is True and kind != "frozen_source_replay":
            raise DataGateError(
                f"replay row lacks frozen replay provenance: {event_id}"
            )
        if replay is False and kind != "fresh_source_trajectory":
            raise DataGateError(f"collection row is not freshly executed: {event_id}")
        expected_split = _expected_source_split(seed)
        if kind == "fresh_source_trajectory" and split != expected_split:
            raise DataGateError(f"source seed/split mismatch: {event_id}")
        required = {
            "game_id": game,
            "seed": seed,
            "split": split,
            "manifest_checksum": expected_manifest,
        }
        for key, value in required.items():
            if provenance.get(key) != value:
                raise DataGateError(f"provenance mismatch for {key}: {event_id}")
        if (
            kind == "fresh_source_trajectory"
            and provenance.get("environment_sha256") != expected_environment
        ):
            raise DataGateError(f"environment provenance mismatch: {event_id}")
        if kind == "frozen_source_replay":
            _validate_replay_provenance(
                provenance,
                manifest=manifest,
                game_id=game,
                event_id=event_id,
            )
            replay_source = (
                str(provenance["source_shard_sha256"]),
                int(provenance["source_line"]),
                str(provenance["arm"]),
            )
            if replay_source in seen_replay_sources:
                raise DataGateError(f"duplicate replay source transition: {event_id}")
            seen_replay_sources.add(replay_source)
        _validate_source_event_schema(
            row,
            event_id=event_id,
            fresh=kind == "fresh_source_trajectory",
        )
        _validate_correspondence_evidence(row, event_id=event_id)
        model_view = row.get("model_view")
        if not isinstance(model_view, Mapping):
            raise DataGateError(f"event lacks a model_view: {event_id}")
        hits = forbidden_model_view_hits(model_view)
        if hits:
            raise FirewallError(
                f"identity-bearing model_view for {event_id}: {', '.join(hits)}"
            )


def _number(
    row: Mapping[str, Any],
    nested_key: str,
    name: str,
    *,
    default: float,
) -> float:
    nested = row.get(nested_key, {})
    if isinstance(nested, Mapping):
        value = nested.get(name, row.get(name, default))
    else:
        value = row.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _boolean(
    row: Mapping[str, Any], nested_key: str, name: str, *, default: bool
) -> bool:
    nested = row.get(nested_key, {})
    value = (
        nested.get(name, row.get(name, default))
        if isinstance(nested, Mapping)
        else row.get(name, default)
    )
    return value is True


def _declared_learned_predicates(row: Mapping[str, Any]) -> frozenset[str]:
    explicit = row.get("learned_predicates")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        return frozenset(
            str(value).casefold()
            for value in explicit
            if str(value).strip()
            and str(value).casefold() not in RARE_EVALUATION_ANCHORS
        )
    metadata = row.get("label_metadata", row.get("labels_metadata", {}))
    if isinstance(metadata, Mapping):
        declared = set()
        for raw_name, specification in metadata.items():
            learned = (
                specification.get("learned") is True
                if isinstance(specification, Mapping)
                else str(specification).casefold() == "learned"
            )
            name = str(raw_name).casefold()
            if learned and name not in RARE_EVALUATION_ANCHORS:
                declared.add(name)
        return frozenset(declared)
    return frozenset()


def build_qa_report(
    *, manifest: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Evaluate every pre-fit provenance and derived-label gate."""

    validate_source_events(events, manifest=manifest, replay=None)
    gate = manifest["qa_gate"]
    # Pool correspondence trials rather than averaging event-level fractions.
    # Otherwise a large, poorly matched projection can be hidden by many tiny
    # one-entity events, despite representing most of the actual evidence.
    correspondence_denominator = sum(
        int(_mapping(row.get("correspondence")).get("fraction_denominator", 0))
        for row in events
    )
    confident_matches = sum(
        int(_mapping(row.get("correspondence")).get("confident_matches", 0))
        for row in events
    )
    fully_ambiguous_matches = sum(
        int(_mapping(row.get("correspondence")).get("fully_ambiguous_matches", 0))
        for row in events
    )
    confident_fraction = (
        confident_matches / correspondence_denominator
        if correspondence_denominator
        else 0.0
    )
    ambiguous_fraction = (
        fully_ambiguous_matches / correspondence_denominator
        if correspondence_denominator
        else 1.0
    )

    exact_transport_certificates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    noncomparable_transport_certificates = 0
    certificate_schema_valid = True
    for row in events:
        certificates = row.get("transport_certificates")
        if not isinstance(certificates, (list, tuple)):
            certificate_schema_valid = False
            continue
        for certificate in certificates:
            if not isinstance(certificate, Mapping) or certificate.get("exact") not in (
                True,
                False,
            ):
                certificate_schema_valid = False
                continue
            if certificate.get("exact") is True:
                if (
                    certificate.get("live_graph_exact_attested") is not True
                    or certificate.get("summary_commutative_exact") is not True
                    or certificate.get("certifies_gauge_equivalence") is not True
                ):
                    certificate_schema_valid = False
                    continue
                exact_transport_certificates.append((row, certificate))
            else:
                if certificate.get("comparable") is not False:
                    certificate_schema_valid = False
                noncomparable_transport_certificates += 1
    transport_round_trip = all(
        certificate.get("round_trip_exact") is True
        and certificate.get("certifies_gauge_equivalence") is True
        for _row, certificate in exact_transport_certificates
    )
    permutation_invariant = all(
        _boolean(row, "transport", "entity_permutation_invariant", default=False)
        for row, _certificate in exact_transport_certificates
    )
    commutative = all(
        isinstance(certificate.get("commutativity"), Mapping)
        and certificate["commutativity"].get("exact") is True
        and certificate.get("summary_commutative_exact") is True
        and _boolean(_row, "transport", "summary_commutative_exact", default=False)
        for _row, certificate in exact_transport_certificates
    )

    predicate_counts: Counter[str] = Counter()
    predicate_totals: Counter[str] = Counter()
    predicate_games: dict[str, set[str]] = defaultdict(set)
    declarations = [_declared_learned_predicates(row) for row in events]
    declared_predicates = (
        frozenset().union(*declarations) if declarations else frozenset()
    )
    declaration_consistent = bool(declarations) and all(
        declaration == declared_predicates and bool(declaration)
        for declaration in declarations
    )
    label_coverage_complete = True
    for row in events:
        labels = row.get("labels", {})
        if not isinstance(labels, Mapping):
            label_coverage_complete = False
            continue
        normalized_labels = {
            str(name).casefold(): value for name, value in labels.items()
        }
        for name in declared_predicates:
            value = normalized_labels.get(name)
            if value not in (True, False, 0, 1):
                label_coverage_complete = False
                continue
            predicate_totals[name] += 1
            if bool(value):
                predicate_counts[name] += 1
                predicate_games[name].add(str(row["game_id"]))

    predicate_metrics: dict[str, dict[str, Any]] = {}
    predicate_checks: dict[str, bool] = {}
    for name in sorted(predicate_totals):
        total = predicate_totals[name]
        positives = predicate_counts[name]
        prevalence = positives / total if total else 0.0
        games = len(predicate_games[name])
        predicate_metrics[name] = {
            "games": games,
            "positives": positives,
            "prevalence": prevalence,
            "total": total,
        }
        predicate_checks[name] = bool(
            float(gate["minimum_predicate_prevalence"])
            <= prevalence
            <= float(gate["maximum_predicate_prevalence"])
            and positives >= int(gate["minimum_predicate_support"])
            and games >= int(gate["minimum_predicate_games"])
        )

    nonterminal = 0
    evaluable = 0
    coherent = 0
    for row in events:
        prefix = row.get("prefix", {})
        if not isinstance(prefix, Mapping):
            prefix = {}
        is_nonterminal = bool(
            prefix.get("nonterminal", row.get("nonterminal_prefix", False))
        )
        if not is_nonterminal:
            continue
        nonterminal += 1
        if bool(prefix.get("evaluable", row.get("evaluable_nonterminal", False))):
            evaluable += 1
        coherent_frames = sum(
            projection.get("complete") is True
            for projection in _mapping(row.get("projections")).values()
            if isinstance(projection, Mapping)
        )
        if coherent_frames >= 2:
            coherent += 1
    evaluable_fraction = evaluable / nonterminal if nonterminal else 0.0
    coherent_fraction = coherent / nonterminal if nonterminal else 0.0

    checks = {
        "events_present": bool(events),
        "persistent_correspondence": confident_fraction
        >= float(gate["minimum_confident_correspondence"]),
        "fully_ambiguous_correspondence": ambiguous_fraction
        < float(gate["maximum_fully_ambiguous_correspondence"]),
        "transport_round_trip_exact": transport_round_trip,
        "entity_permutation_invariant": permutation_invariant,
        "transport_commutative_exact": commutative,
        "transport_certificate_schema": certificate_schema_valid,
        "exact_transport_evidence_present": bool(exact_transport_certificates),
        "learned_predicates_present": bool(declared_predicates),
        "learned_predicate_declaration_consistent": declaration_consistent,
        "learned_predicate_label_coverage": label_coverage_complete,
        "learned_predicate_prevalence_and_support": bool(predicate_checks)
        and all(predicate_checks.values()),
        "evaluable_nonterminal_prefixes": evaluable_fraction
        >= float(gate["minimum_evaluable_nonterminal_prefix_fraction"]),
        "multiframe_coherent_prefixes": coherent_fraction
        >= float(gate["minimum_multiframe_coherent_prefix_fraction"]),
        "holdout_closed": True,
        "ar25_closed": True,
        "source_validation_closed": True,
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "compile",
        "status": "PASS_T10_2_QA" if passed else "DATA_OR_PROVENANCE_INVALID",
        "manifest_checksum": manifest["manifest_checksum"],
        "event_count": len(events),
        "event_ids_sha256": canonical_sha256(sorted(_event_id(row) for row in events)),
        "metrics": {
            "confident_correspondence_fraction": confident_fraction,
            "fully_ambiguous_correspondence_fraction": ambiguous_fraction,
            "correspondence_trials": correspondence_denominator,
            "confident_correspondence_matches": confident_matches,
            "fully_ambiguous_correspondence_matches": fully_ambiguous_matches,
            "nonterminal_prefixes": nonterminal,
            "evaluable_nonterminal_prefix_fraction": evaluable_fraction,
            "multiframe_coherent_prefix_fraction": coherent_fraction,
            "predicates": predicate_metrics,
            "declared_learned_predicates": sorted(declared_predicates),
            "exact_or_invertible_transports": len(exact_transport_certificates),
            "partial_noncomparable_transports": (noncomparable_transport_certificates),
        },
        "checks": checks,
        "passed": passed,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    return signed_payload(payload, checksum_key="report_checksum")


def _invoke_with_context(
    function: Callable[..., Any],
    *,
    context: Mapping[str, Any],
    positional_fallback: Sequence[Any] = (),
) -> Any:
    """Call an injected adapter without hiding exceptions raised inside it."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*positional_fallback)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepted = {
        name: value
        for name, value in context.items()
        if accepts_kwargs or name in parameters
    }
    required_unknown = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.name not in accepted
    ]
    if required_unknown:
        return function(*positional_fallback[: len(required_unknown)])
    return function(**accepted)


def _environment_rows(environment: Any, *, context: Mapping[str, Any]) -> list[Any]:
    if hasattr(environment, "collect_events"):
        result = _invoke_with_context(
            environment.collect_events,
            context=context,
            positional_fallback=(
                context["seed"],
                context["resets"],
                context["action_budget"],
            ),
        )
    elif hasattr(environment, "run_validation"):
        result = _invoke_with_context(
            environment.run_validation,
            context=context,
            positional_fallback=(context["seed"],),
        )
    elif callable(environment):
        result = _invoke_with_context(
            environment,
            context=context,
            positional_fallback=(context["seed"],),
        )
    else:
        result = environment
    if isinstance(result, Mapping) and "events" in result:
        result = result["events"]
    if isinstance(result, Mapping):
        return [dict(result)]
    if isinstance(result, (str, bytes)) or not isinstance(result, Iterable):
        raise RuntimeUnavailableError(
            "environment adapter must return an event/result iterable"
        )
    return list(result)


def _fresh_source_event(
    raw: Any,
    *,
    game_id: str,
    seed: int,
    split: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        to_dict = getattr(raw, "to_dict", None)
        raw = to_dict() if callable(to_dict) else None
    if not isinstance(raw, Mapping):
        raise DataGateError("environment produced a non-mapping physical event")
    event = dict(raw)
    event_game = str(event.setdefault("game_id", game_id))
    event_seed = int(event.setdefault("seed", seed))
    event_split = str(event.setdefault("split", split))
    if (event_game, event_seed, event_split) != (game_id, seed, split):
        raise DataGateError("environment event escaped its registered source lane")
    provenance = dict(event.get("provenance", {}))
    provenance.update(
        {
            "kind": "fresh_source_trajectory",
            "game_id": game_id,
            "seed": seed,
            "split": split,
            "manifest_checksum": manifest["manifest_checksum"],
            "environment_sha256": manifest["environment_sha256"],
        }
    )
    event["provenance"] = provenance
    return seal_event(event)


def _call_env_factory(
    env_factory: Callable[..., Any],
    *,
    game_id: str,
    seed: int,
    phase: str,
    split: str | None = None,
    held_out_game: str | None = None,
) -> Any:
    return _invoke_with_context(
        env_factory,
        context={
            "game_id": game_id,
            "seed": seed,
            "phase": phase,
            "split": split,
            "held_out_game": held_out_game,
            "training_games": tuple(
                source for source in SOURCE_GAMES if source != held_out_game
            ),
        },
        positional_fallback=(game_id, seed),
    )


def _default_env_factory(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeUnavailableError(
        "no T10.2 live runtime is configured; inject env_factory explicitly"
    )


_CROSS_FIT_UNIT_FIELDS = frozenset(
    {
        "held_out_game",
        "seed",
        "training_games",
        "donor_event_count",
        "donor_event_ids_sha256",
        "held_out_prefit_events_used",
        "resets",
    }
)
_CROSS_FIT_RESET_FIELDS = frozenset(
    {
        "reset_index",
        "controller",
        "action_count",
        "online_observations",
        "error_count",
        "initial_particle_count",
        "initial_class_count",
        "final_particle_count",
        "final_class_count",
        "stop_reason",
    }
)


def _source_factory_code_binding(
    factory: Any,
    *,
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    candidate = type(factory)
    try:
        source_file = inspect.getsourcefile(candidate)
    except (OSError, TypeError):
        source_file = None
    expected_path = (repo_root / "theory/sage_t/t10_2_runtime.py").resolve()
    observed_path = Path(source_file).resolve() if source_file else None
    expected_digest = _mapping(manifest.get("code_sha256")).get(
        "theory/sage_t/t10_2_runtime.py"
    )
    observed_digest = (
        file_sha256(observed_path)
        if observed_path is not None and observed_path.is_file()
        else ""
    )
    manifest_checksum = str(getattr(factory, "manifest_checksum", ""))
    code_bound = bool(
        candidate.__module__ == "theory.sage_t.t10_2_runtime"
        and candidate.__name__ == "T10_2SourceFactory"
        and observed_path == expected_path
        and isinstance(expected_digest, str)
        and observed_digest == expected_digest
        and manifest_checksum == manifest.get("manifest_checksum")
    )
    return {
        "module": candidate.__module__,
        "class": candidate.__name__,
        "source_sha256": observed_digest,
        "manifest_checksum": manifest_checksum,
        "code_bound": code_bound,
    }


def _expected_cross_fit_resets(seed: int) -> tuple[str, ...]:
    forward = (
        "learned",
        "capacity_matched_independent",
        "capacity_matched_independent",
        "learned",
    )
    return forward if int(seed) % 2 == 0 else tuple(reversed(forward))


def _fallback_cross_fit_units(
    source_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Represent every registered unit even for a test-only unverified factory."""

    units: list[dict[str, Any]] = []
    discovery = [row for row in source_events if row.get("split") == "discovery"]
    for held_out_game in SOURCE_GAMES:
        donors = tuple(game for game in SOURCE_GAMES if game != held_out_game)
        donor_ids = [
            _event_id(row) for row in discovery if str(row.get("game_id", "")) in donors
        ]
        for seed in CONFIRMATION_SEEDS:
            resets: list[dict[str, Any]] = []
            for reset_index, controller in enumerate(_expected_cross_fit_resets(seed)):
                matching = [
                    row
                    for row in source_events
                    if row.get("split") == "leave_one_game_out_confirmation"
                    and str(row.get("game_id", "")) == held_out_game
                    and int(row.get("seed", -1)) == seed
                    and isinstance(row.get("selection"), Mapping)
                    and int(row["selection"].get("reset_index", -1)) == reset_index
                    and row["selection"].get("controller") == controller
                ]
                resets.append(
                    {
                        "reset_index": reset_index,
                        "controller": controller,
                        "action_count": len(matching),
                        "online_observations": 0,
                        "error_count": 0,
                        "initial_particle_count": 0,
                        "initial_class_count": 0,
                        "final_particle_count": 0,
                        "final_class_count": 0,
                        "stop_reason": "unverified_factory",
                    }
                )
            units.append(
                {
                    "held_out_game": held_out_game,
                    "seed": seed,
                    "training_games": list(donors),
                    "donor_event_count": len(donor_ids),
                    "donor_event_ids_sha256": canonical_sha256(donor_ids),
                    "held_out_prefit_events_used": 0,
                    "resets": resets,
                }
            )
    return units


def _canonical_cross_fit_units(units: Any) -> list[dict[str, Any]]:
    if not isinstance(units, (list, tuple)):
        raise DataGateError("cross-fit audit units must be a sequence")
    canonical: list[dict[str, Any]] = []
    for raw_unit in units:
        if not isinstance(raw_unit, Mapping) or set(raw_unit) != set(
            _CROSS_FIT_UNIT_FIELDS
        ):
            raise DataGateError("cross-fit audit unit schema drifted")
        raw_resets = raw_unit.get("resets")
        if not isinstance(raw_resets, (list, tuple)):
            raise DataGateError("cross-fit audit resets must be a sequence")
        resets: list[dict[str, Any]] = []
        for raw_reset in raw_resets:
            if not isinstance(raw_reset, Mapping) or set(raw_reset) != set(
                _CROSS_FIT_RESET_FIELDS
            ):
                raise DataGateError("cross-fit audit reset schema drifted")
            integer_fields = (
                "reset_index",
                "action_count",
                "online_observations",
                "error_count",
                "initial_particle_count",
                "initial_class_count",
                "final_particle_count",
                "final_class_count",
            )
            normalized: dict[str, Any] = {}
            for field in integer_fields:
                value = _strict_nonnegative_int(raw_reset.get(field))
                if value is None:
                    raise DataGateError(f"cross-fit audit has invalid {field}")
                normalized[field] = value
            controller = str(raw_reset.get("controller", ""))
            stop_reason = str(raw_reset.get("stop_reason", ""))
            if controller not in {"learned", "capacity_matched_independent"}:
                raise DataGateError("cross-fit audit has an invalid controller")
            if not stop_reason:
                raise DataGateError("cross-fit audit lacks a stop reason")
            resets.append(
                {
                    **normalized,
                    "controller": controller,
                    "stop_reason": stop_reason,
                }
            )
        seed = _strict_nonnegative_int(raw_unit.get("seed"))
        donor_count = _strict_nonnegative_int(raw_unit.get("donor_event_count"))
        held_out_used = _strict_nonnegative_int(
            raw_unit.get("held_out_prefit_events_used")
        )
        training_games = raw_unit.get("training_games")
        if (
            seed is None
            or donor_count is None
            or held_out_used is None
            or not isinstance(training_games, (list, tuple))
        ):
            raise DataGateError("cross-fit audit unit values are invalid")
        donor_digest = str(raw_unit.get("donor_event_ids_sha256", ""))
        if not _SHA256.fullmatch(donor_digest):
            raise DataGateError("cross-fit audit donor digest is invalid")
        canonical.append(
            {
                "held_out_game": str(raw_unit.get("held_out_game", "")),
                "seed": seed,
                "training_games": [str(value) for value in training_games],
                "donor_event_count": donor_count,
                "donor_event_ids_sha256": donor_digest,
                "held_out_prefit_events_used": held_out_used,
                "resets": sorted(resets, key=lambda row: row["reset_index"]),
            }
        )
    return sorted(canonical, key=lambda row: (row["held_out_game"], row["seed"]))


def _cross_fit_audit_checks(
    *,
    manifest: Mapping[str, Any],
    source_events: Sequence[Mapping[str, Any]],
    factory: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    expected_runtime_digest = _mapping(manifest.get("code_sha256")).get(
        "theory/sage_t/t10_2_runtime.py"
    )
    expected_units = {
        (game, seed) for game in SOURCE_GAMES for seed in CONFIRMATION_SEEDS
    }
    observed_units = {
        (str(unit.get("held_out_game", "")), int(unit.get("seed", -1)))
        for unit in units
    }
    exact_donors = True
    exact_resets = True
    ledger_counts_match = True
    online_only = True
    held_out_closed = True
    donor_binding = True
    effective_capacity_by_fold = True
    every_unit_observed = True
    execution_error_free = True
    discovery = [row for row in source_events if row.get("split") == "discovery"]
    for unit in units:
        game = str(unit.get("held_out_game", ""))
        seed = int(unit.get("seed", -1))
        donors = tuple(source for source in SOURCE_GAMES if source != game)
        exact_donors = exact_donors and tuple(unit.get("training_games", ())) == donors
        donor_ids = [
            _event_id(row) for row in discovery if str(row.get("game_id", "")) in donors
        ]
        donor_binding = donor_binding and bool(
            int(unit.get("donor_event_count", -1)) == len(donor_ids)
            and unit.get("donor_event_ids_sha256") == canonical_sha256(donor_ids)
        )
        held_out_closed = (
            held_out_closed and int(unit.get("held_out_prefit_events_used", -1)) == 0
        )
        resets = tuple(unit.get("resets", ()))
        expected_controllers = _expected_cross_fit_resets(seed)
        exact_resets = exact_resets and bool(
            len(resets) == SOURCE_RESETS_PER_GAME_SEED
            and tuple(int(row.get("reset_index", -1)) for row in resets)
            == tuple(range(SOURCE_RESETS_PER_GAME_SEED))
            and tuple(str(row.get("controller", "")) for row in resets)
            == expected_controllers
        )
        controller_actions = {"learned": 0, "capacity_matched_independent": 0}
        capacities: dict[str, set[tuple[int, int]]] = {
            "learned": set(),
            "capacity_matched_independent": set(),
        }
        for reset in resets:
            reset_index = int(reset.get("reset_index", -1))
            controller = str(reset.get("controller", ""))
            matching = [
                row
                for row in source_events
                if row.get("split") == "leave_one_game_out_confirmation"
                and str(row.get("game_id", "")) == game
                and int(row.get("seed", -1)) == seed
                and isinstance(row.get("selection"), Mapping)
                and int(row["selection"].get("reset_index", -1)) == reset_index
                and row["selection"].get("controller") == controller
            ]
            action_count = int(reset.get("action_count", -1))
            online_count = int(reset.get("online_observations", -1))
            execution_error_free = (
                execution_error_free and int(reset.get("error_count", -1)) == 0
            )
            ledger_counts_match = ledger_counts_match and action_count == len(matching)
            online_only = online_only and online_count == action_count
            if controller in controller_actions:
                controller_actions[controller] += max(0, action_count)
                capacities[controller].add(
                    (
                        int(reset.get("initial_particle_count", 0)),
                        int(reset.get("initial_class_count", 0)),
                    )
                )
        every_unit_observed = every_unit_observed and all(
            value > 0 for value in controller_actions.values()
        )
        learned_capacity = capacities["learned"]
        independent_capacity = capacities["capacity_matched_independent"]
        effective_capacity_by_fold = effective_capacity_by_fold and bool(
            len(learned_capacity) == 1
            and learned_capacity == independent_capacity
            and next(iter(learned_capacity), (0, 0))[0] > 0
            and next(iter(learned_capacity), (0, 0))[1] > 0
        )
    return {
        "factory_code_bound": bool(
            factory.get("code_bound") is True
            and factory.get("module") == "theory.sage_t.t10_2_runtime"
            and factory.get("class") == "T10_2SourceFactory"
            and isinstance(expected_runtime_digest, str)
            and factory.get("source_sha256") == expected_runtime_digest
            and factory.get("manifest_checksum") == manifest.get("manifest_checksum")
        ),
        "exact_nine_units": bool(
            len(units) == len(expected_units) and observed_units == expected_units
        ),
        "exact_two_resets_per_arm": exact_resets,
        "exact_leave_one_game_out_donors": exact_donors,
        "donor_events_bound": donor_binding,
        "held_out_prefit_zero": held_out_closed,
        "ledger_action_counts_match": ledger_counts_match,
        "online_observations_only_from_executed_actions": online_only,
        "effective_capacity_matched_by_fold": effective_capacity_by_fold,
        "every_unit_has_both_arms_observed": every_unit_observed,
        "execution_error_free": execution_error_free,
    }


def build_cross_fit_audit(
    *,
    manifest: Mapping[str, Any],
    source_event_path: str | Path,
    source_events: Sequence[Mapping[str, Any]],
    units: Any,
    factory_binding: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_units = _canonical_cross_fit_units(units)
    factory = dict(factory_binding)
    checks = _cross_fit_audit_checks(
        manifest=manifest,
        source_events=source_events,
        factory=factory,
        units=canonical_units,
    )
    payload = {
        "format_version": CROSS_FIT_AUDIT_FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "source_events": artifact_descriptor(source_event_path),
        "source_event_ids_sha256": canonical_sha256(
            [_event_id(row) for row in source_events]
        ),
        "factory": factory,
        "registered_unit_count": len(canonical_units),
        "units": canonical_units,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return signed_payload(payload, checksum_key="audit_checksum")


def read_cross_fit_audit(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
    source_event_path: str | Path,
    source_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = read_checked_json(path, checksum_key="audit_checksum")
    expected_fields = {
        "format_version",
        "manifest_checksum",
        "source_events",
        "source_event_ids_sha256",
        "factory",
        "registered_unit_count",
        "units",
        "checks",
        "passed",
        "audit_checksum",
    }
    if set(audit) != expected_fields:
        raise ManifestDriftError("cross-fit audit schema drifted")
    if (
        audit.get("format_version") != CROSS_FIT_AUDIT_FORMAT_VERSION
        or audit.get("manifest_checksum") != manifest.get("manifest_checksum")
        or audit.get("source_events") != artifact_descriptor(source_event_path)
        or audit.get("source_event_ids_sha256")
        != canonical_sha256([_event_id(row) for row in source_events])
    ):
        raise ManifestDriftError("cross-fit audit binding drifted")
    factory = audit.get("factory")
    if not isinstance(factory, Mapping) or set(factory) != {
        "module",
        "class",
        "source_sha256",
        "manifest_checksum",
        "code_bound",
    }:
        raise ManifestDriftError("cross-fit audit factory binding drifted")
    units = _canonical_cross_fit_units(audit.get("units"))
    if units != audit.get("units") or audit.get("registered_unit_count") != len(units):
        raise ManifestDriftError("cross-fit audit unit registry drifted")
    checks = _cross_fit_audit_checks(
        manifest=manifest,
        source_events=source_events,
        factory=factory,
        units=units,
    )
    if audit.get("checks") != checks or audit.get("passed") is not all(checks.values()):
        raise ManifestDriftError("cross-fit audit checks drifted")
    return audit


def _source_lane_action_budget(collected_actions: int) -> int:
    if isinstance(collected_actions, bool) or collected_actions < 0:
        raise DataGateError("source collection action accounting is invalid")
    remaining = SOURCE_MAXIMUM_NEW_ACTIONS - collected_actions
    if remaining < 0:
        raise DataGateError("source collection exceeded 4,607 new actions")
    return min(SOURCE_RESETS_PER_GAME_SEED * SOURCE_ACTIONS_PER_RESET, remaining)


def collect_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
    env_factory: Callable[..., Any] | None = None,
    games: Sequence[str] | None = None,
    resource_probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    clock: Callable[[], float] = time.perf_counter,
    _test_only_allow_factory: bool = False,
    _test_only_allow_clock: bool = False,
) -> dict[str, Any]:
    """Collect only the frozen source lanes through an injected environment."""

    if clock is not time.perf_counter and not _test_only_allow_clock:
        raise RuntimeUnavailableError(
            "production collection requires the code-bound monotonic clock"
        )
    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    selected_games = tuple(games or manifest["source_plan"]["games"])

    # Every requested target is checked before resource setup or factory use.
    for game_id in selected_games:
        enforce_environment_firewall(phase="collect", game_id=game_id)
    if selected_games != SOURCE_GAMES:
        raise FirewallError("collection must use the exact frozen source allowlist")

    snapshot = guard_resources(
        repo_root=root,
        limits=limits,
        expensive=True,
        probe=resource_probe,
    )
    factory = env_factory or _default_env_factory
    factory_binding = _source_factory_code_binding(
        factory,
        manifest=manifest,
        repo_root=root,
    )
    if factory_binding["code_bound"] is not True and not _test_only_allow_factory:
        raise RuntimeUnavailableError(
            "production collection requires the manifest-bound T10.2 source factory"
        )
    rows: list[dict[str, Any]] = []
    previous_clock: float | None = None

    def sample_clock(label: str) -> float:
        nonlocal previous_clock
        try:
            observed = float(clock())
        except (TypeError, ValueError) as exc:
            raise DataGateError(f"collection monotonic {label} is invalid") from exc
        if not math.isfinite(observed) or (
            previous_clock is not None and observed < previous_clock
        ):
            raise DataGateError(f"collection monotonic {label} regressed")
        previous_clock = observed
        return observed

    started = sample_clock("start")
    lanes = [
        ("discovery", game_id, seed, None)
        for game_id in selected_games
        for seed in DISCOVERY_SEEDS
    ]
    lanes.extend(
        (
            "leave_one_game_out_confirmation",
            held_out_game,
            seed,
            held_out_game,
        )
        for held_out_game in selected_games
        for seed in CONFIRMATION_SEEDS
    )
    for split, game_id, seed, held_out_game in lanes:
        if sample_clock("lane preflight") - started > SOURCE_MAXIMUM_WALL_SECONDS:
            raise ResourceGateError("source collection exceeded 5,400 seconds")
        enforce_environment_firewall(phase="collect", game_id=game_id)
        environment = _call_env_factory(
            factory,
            game_id=game_id,
            seed=seed,
            phase="collect",
            split=split,
            held_out_game=held_out_game,
        )
        try:
            total_action_budget = _source_lane_action_budget(len(rows))
            raw_rows = _environment_rows(
                environment,
                context={
                    "game_id": game_id,
                    "seed": seed,
                    "split": split,
                    "held_out_game": held_out_game,
                    "training_games": tuple(
                        source for source in SOURCE_GAMES if source != held_out_game
                    ),
                    "resets": SOURCE_RESETS_PER_GAME_SEED,
                    "action_budget": SOURCE_ACTIONS_PER_RESET,
                    "total_action_budget": total_action_budget,
                    "stop_on_progress": True,
                    "stop_on_game_over": True,
                },
            )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        if len(raw_rows) > total_action_budget:
            raise DataGateError(f"source lane exceeded action budget: {game_id}/{seed}")
        rows.extend(
            _fresh_source_event(
                raw,
                game_id=game_id,
                seed=seed,
                split=split,
                manifest=manifest,
            )
            for raw in raw_rows
        )
        guard_resources(
            repo_root=root,
            limits=limits,
            expensive=True,
            probe=resource_probe,
        )
        if sample_clock("lane completion") - started > SOURCE_MAXIMUM_WALL_SECONDS:
            raise ResourceGateError("source collection exceeded 5,400 seconds")
    if len(rows) > SOURCE_MAXIMUM_NEW_ACTIONS:
        raise DataGateError("source collection exceeded 4,607 new actions")
    validate_source_events(rows, manifest=manifest, replay=False)
    if sample_clock("ledger preflight") - started > SOURCE_MAXIMUM_WALL_SECONDS:
        raise ResourceGateError("source collection exceeded 5,400 seconds")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "source_events.jsonl"
    write_event_ledger(ledger_path, rows)
    enforce_artifact_limit(ledger_path, kind="ledger", limits=limits)
    runtime_units = getattr(factory, "cross_fit_audit", ())
    units = (
        runtime_units
        if isinstance(runtime_units, (list, tuple)) and runtime_units
        else _fallback_cross_fit_units(rows)
    )
    cross_fit_audit = build_cross_fit_audit(
        manifest=manifest,
        source_event_path=ledger_path,
        source_events=rows,
        units=units,
        factory_binding=factory_binding,
    )
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    write_compact_json(cross_fit_path, cross_fit_audit)
    enforce_artifact_limit(cross_fit_path, kind="derived", limits=limits)
    report_path = destination / "collection_report.json"

    def persist_collection_report(finished: float) -> dict[str, Any]:
        wall_seconds = finished - started
        if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise DataGateError("collection monotonic elapsed duration is invalid")
        payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "phase": "collect",
            "status": "T10_2_SOURCE_COLLECTION_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "games": list(selected_games),
            "splits": {
                "discovery": list(DISCOVERY_SEEDS),
                "leave_one_game_out_confirmation": list(CONFIRMATION_SEEDS),
            },
            "event_count": len(rows),
            "precollection_aborted_actions": (SOURCE_PRECOLLECTION_ABORTED_ACTIONS),
            "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
            "accounted_action_count": (
                SOURCE_PRECOLLECTION_ABORTED_ACTIONS + len(rows)
            ),
            "wall_seconds": wall_seconds,
            "timing": {
                "clock": "time.perf_counter",
                "monotonic_started": started,
                "monotonic_finished": finished,
                "monotonic_elapsed_seconds": wall_seconds,
            },
            "events": artifact_descriptor(ledger_path),
            "cross_fit_audit": artifact_descriptor(cross_fit_path),
            "cross_fit_checks": cross_fit_audit["checks"],
            "resource_preflight": snapshot.to_dict(),
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
            },
        }
        persisted = signed_payload(payload, checksum_key="report_checksum")
        write_compact_json(report_path, persisted)
        enforce_output_artifacts(destination, limits=limits)
        return persisted

    preliminary_finished = sample_clock("preliminary finish")
    persist_collection_report(preliminary_finished)
    collection_finished = sample_clock("persistence finish")
    report = persist_collection_report(collection_finished)
    post_persistence_elapsed = sample_clock("post-persistence recheck") - started
    if post_persistence_elapsed > SOURCE_MAXIMUM_WALL_SECONDS:
        raise ResourceGateError(
            "source collection exceeded 5,400 seconds after persistence"
        )
    return report


def _verify_artifact_binding(
    report: Mapping[str, Any], path: Path, *, key: str
) -> None:
    expected = report.get(key)
    if not isinstance(expected, Mapping) or dict(expected) != artifact_descriptor(path):
        raise ManifestDriftError(f"artifact binding drifted: {path}")


def compile_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    event_path: str | Path | None = None,
    collection_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    resource_probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    guard_resources(
        repo_root=root,
        limits=limits,
        expensive=False,
        probe=resource_probe,
    )
    destination = Path(output_dir)
    source_path = Path(event_path or destination / "source_events.jsonl")
    collection_path = Path(
        collection_report_path or destination / "collection_report.json"
    )
    collection = read_checked_json(collection_path)
    if collection.get("status") != "T10_2_SOURCE_COLLECTION_COMPLETE":
        raise GateRefusalError("source collection is incomplete")
    if collection.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("collection/manifest binding drifted")
    if (
        collection.get("precollection_aborted_actions")
        != SOURCE_PRECOLLECTION_ABORTED_ACTIONS
        or collection.get("maximum_new_actions") != SOURCE_MAXIMUM_NEW_ACTIONS
    ):
        raise ManifestDriftError("collection source action budget binding drifted")
    _verify_artifact_binding(collection, source_path, key="events")
    events = read_event_ledger(source_path)
    if (
        collection.get("event_count") != len(events)
        or len(events) > SOURCE_MAXIMUM_NEW_ACTIONS
        or collection.get("accounted_action_count")
        != SOURCE_PRECOLLECTION_ABORTED_ACTIONS + len(events)
    ):
        raise ManifestDriftError("collection source action accounting drifted")
    validate_source_events(events, manifest=manifest, replay=False)
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    _verify_artifact_binding(collection, cross_fit_path, key="cross_fit_audit")
    cross_fit_audit = read_cross_fit_audit(
        cross_fit_path,
        manifest=manifest,
        source_event_path=source_path,
        source_events=events,
    )
    fresh_qa = build_qa_report(manifest=manifest, events=events)
    report = signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "phase": "compile",
            "status": "T10_2_FRESH_INTEGRITY_COMPLETE",
            "manifest_checksum": manifest["manifest_checksum"],
            "integrity_passed": True,
            "passed": True,
            "checks": {
                "source_event_schema": True,
                "source_provenance": True,
                "event_checksums": True,
                "source_firewall": True,
                "unique_event_ids": True,
                "cross_fit_audit_bound": True,
            },
            "fresh_scientific_qa": {
                "status": fresh_qa["status"],
                "passed": fresh_qa["passed"],
                "checks": fresh_qa["checks"],
                "metrics": fresh_qa["metrics"],
                "report_checksum": fresh_qa["report_checksum"],
            },
            "inputs": {
                "source_events": artifact_descriptor(source_path),
                "cross_fit_audit": artifact_descriptor(cross_fit_path),
            },
            "cross_fit_checks": cross_fit_audit["checks"],
            "evidence_scope": (
                "fresh integrity and provenance only; scientific QA becomes "
                "blocking on fresh+replay immediately before source fit"
            ),
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_compact_json(destination / "compile_report.json", report)
    enforce_output_artifacts(destination, limits=limits)
    return report


def replay_phase(
    *,
    replay_input_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    compile_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    resource_probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    """Admit a frozen source-only replay ledger after full provenance checks."""

    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    guard_resources(
        repo_root=root,
        limits=limits,
        expensive=False,
        probe=resource_probe,
    )
    destination = Path(output_dir)
    compile_report = read_checked_json(
        compile_report_path or destination / "compile_report.json"
    )
    if not _compile_integrity_passed(compile_report):
        raise GateRefusalError("replay is blocked until fresh integrity passes")
    if compile_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("compile/manifest binding drifted")
    rows = read_event_ledger(replay_input_path)
    validate_source_events(rows, manifest=manifest, replay=True)
    output_path = destination / "replay_events.jsonl"
    write_event_ledger(output_path, rows)
    enforce_artifact_limit(output_path, kind="ledger", limits=limits)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "replay",
        "status": "T10_2_SOURCE_REPLAY_COMPLETE",
        "manifest_checksum": manifest["manifest_checksum"],
        "event_count": len(rows),
        "input": artifact_descriptor(replay_input_path),
        "events": artifact_descriptor(output_path),
        "checks": {
            "source_only": True,
            "complete_provenance": True,
            "unique_event_ids": True,
            "holdout_closed": True,
        },
    }
    report = signed_payload(payload, checksum_key="report_checksum")
    write_compact_json(destination / "replay_report.json", report)
    enforce_output_artifacts(destination, limits=limits)
    return report


def validate_prefit_evidence(
    *,
    manifest: Mapping[str, Any],
    fresh_event_path: str | Path,
    replay_event_path: str | Path,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Re-read fresh and replay ledgers together immediately before fitting."""

    fresh_path = Path(fresh_event_path)
    replay_path = Path(replay_event_path)
    fresh = read_event_ledger(fresh_path)
    replay = read_event_ledger(replay_path)
    if not fresh or not replay:
        raise DataGateError("source control evidence requires both nonempty ledgers")
    validate_source_events(fresh, manifest=manifest, replay=False)
    validate_source_events(replay, manifest=manifest, replay=True)
    # The combined pass catches an event duplicated across evidence sources.
    combined = [*fresh, *replay]
    validate_source_events(combined, manifest=manifest, replay=None)
    combined_qa = build_qa_report(manifest=manifest, events=combined)
    result = {
        "fresh_events": len(fresh),
        "replay_events": len(replay),
        "combined_events": len(combined),
        "fresh_ledger": artifact_descriptor(fresh_path),
        "replay_ledger": artifact_descriptor(replay_path),
        "revalidated_before_fit": True,
        "combined_qa_status": combined_qa["status"],
        "combined_qa_passed": combined_qa["passed"],
        "combined_qa_report_checksum": combined_qa["report_checksum"],
        "combined_qa_metrics": combined_qa["metrics"],
        "combined_qa_checks": combined_qa["checks"],
    }
    if require_pass and combined_qa.get("status") != "PASS_T10_2_QA":
        raise DataGateError("combined fresh+replay QA failed before fit")
    return result


def build_data_invalid_source_report(
    *,
    manifest: Mapping[str, Any],
    reason: str,
    compile_report: Mapping[str, Any] | None = None,
    prefit_evidence: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the terminal, validation-closed report for a pre-fit QA stop."""

    compile_passed = compile_report is not None and _compile_integrity_passed(
        compile_report
    )
    combined_passed = (
        prefit_evidence is not None
        and prefit_evidence.get("combined_qa_status") == "PASS_T10_2_QA"
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "source-train",
        "status": "DATA_OR_PROVENANCE_INVALID",
        "verdict": "DATA_OR_PROVENANCE_INVALID",
        "manifest_checksum": manifest["manifest_checksum"],
        "precollection_aborted_actions": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
        "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
        "reason": str(reason),
        "checks": {
            "compile_integrity_passed": compile_passed,
            "combined_prefit_qa_passed": combined_passed,
            "trainer_invoked": False,
            "source_validation_closed": True,
            "ar25_closed": True,
            "holdout_closed": True,
        },
        "compile_report_checksum": (
            None if compile_report is None else compile_report.get("report_checksum")
        ),
        "prefit_evidence": dict(prefit_evidence or {}),
        "inputs": dict(inputs or {}),
        "passed": False,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    return signed_payload(payload, checksum_key="report_checksum")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _compile_integrity_passed(report: Mapping[str, Any]) -> bool:
    """Accept the current integrity status and legacy all-fresh QA reports."""

    return bool(
        (
            report.get("status") == "T10_2_FRESH_INTEGRITY_COMPLETE"
            and report.get("integrity_passed") is True
        )
        or report.get("status") == "PASS_T10_2_QA"
    )


def _strict_control_result(
    control_results: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    result = control_results.get(name)
    if not isinstance(result, Mapping):
        raise DataGateError(f"missing structured source control result: {name}")
    for field in (
        "attempted",
        "execution_ok",
        "scientific_pass",
        "completed",
        "passed",
    ):
        if type(result.get(field)) is not bool:
            raise DataGateError(f"source control lacks strict boolean {field}: {name}")
    if result.get("completed") is not (
        result.get("attempted") is True and result.get("execution_ok") is True
    ) or result.get("passed") is not (result.get("scientific_pass") is True):
        raise DataGateError(f"source control compatibility outcome drifted: {name}")
    return result


def _finite_control_degradation(result: Mapping[str, Any], *, name: str) -> float:
    value = result.get("degradation")
    if isinstance(value, bool):
        raise DataGateError(f"source control lacks finite degradation: {name}")
    try:
        degradation = float(value)
    except (TypeError, ValueError) as exc:
        raise DataGateError(f"source control lacks finite degradation: {name}") from exc
    if not math.isfinite(degradation):
        raise DataGateError(f"source control lacks finite degradation: {name}")
    return degradation


def _strict_positive_count(result: Mapping[str, Any], field: str) -> bool:
    value = result.get(field)
    return bool(isinstance(value, int) and not isinstance(value, bool) and value > 0)


def _derive_source_control_views(
    metrics: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Derive gate-facing views from the exhaustive code-produced evidence."""

    control_results = metrics.get("control_results")
    if not isinstance(control_results, Mapping) or set(control_results) != set(
        REGISTERED_SOURCE_CONTROLS
    ):
        raise DataGateError("source control_results are not the exhaustive registry")
    results = {
        name: _strict_control_result(control_results, name)
        for name in REGISTERED_SOURCE_CONTROLS
    }
    completed = {
        name: bool(
            results[name].get("attempted") is True
            and results[name].get("execution_ok") is True
        )
        for name in REGISTERED_SOURCE_CONTROLS
    }
    attempted = {
        name: results[name].get("attempted") is True
        for name in REGISTERED_SOURCE_CONTROLS
    }
    execution_ok = {
        name: results[name].get("execution_ok") is True
        for name in REGISTERED_SOURCE_CONTROLS
    }

    def passed(name: str) -> bool:
        result = results[name]
        return bool(
            result.get("completed") is True
            and result.get("passed") is True
            and result.get("scientific_pass") is True
        )

    transport = results["transport_oracle"]
    transport_proof_complete = all(
        _strict_positive_count(transport, field)
        for field in (
            "nontrivial_exact_commutative_certificate_count",
            "certified_orbit_witness_candidate_count",
            "posterior_merged_gauge_class_count",
        )
    )
    controls: dict[str, Any] = {
        "no_transport_degradation": _finite_control_degradation(
            results["no_transport"], name="no_transport"
        ),
        "binding_swap_degradation": _finite_control_degradation(
            results["binding_swap"], name="binding_swap"
        ),
        "capacity_matched_independent_posterior_passed": passed(
            "capacity_matched_independent_posterior"
        ),
        "transport_oracle_passed": passed("transport_oracle")
        and transport_proof_complete,
        "dynamics_oracle_passed": passed("dynamics_oracle"),
        "goal_oracle_passed": passed("goal_oracle"),
        "best_executed_sequence_oracle_passed": passed("best_executed_sequence_oracle"),
        "option_oracle_passed": passed("option_oracle"),
        "complete_program_oracle_passed": passed("complete_program_oracle"),
    }
    declared_completed = metrics.get("completed_controls")
    declared_attempted = metrics.get("attempted_controls")
    declared_execution_ok = metrics.get("execution_ok_controls")
    declared_controls = metrics.get("controls")
    if declared_attempted != attempted:
        raise DataGateError(
            "attempted_controls disagree with exhaustive control_results"
        )
    if declared_execution_ok != execution_ok:
        raise DataGateError(
            "execution_ok_controls disagree with exhaustive control_results"
        )
    if declared_completed != completed:
        raise DataGateError(
            "completed_controls disagree with exhaustive control_results"
        )
    if declared_controls != controls:
        raise DataGateError("controls disagree with exhaustive control_results")
    return completed, controls


def _source_evidence_binding(
    *,
    manifest: Mapping[str, Any],
    fresh_path: Path,
    replay_path: Path,
    cross_fit_path: Path,
    fresh: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    control_results: Mapping[str, Any],
    cross_fit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": "sage-t10.2-source-control-evidence-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "fresh_events": artifact_descriptor(fresh_path),
        "replay_events": artifact_descriptor(replay_path),
        "cross_fit_audit": artifact_descriptor(cross_fit_path),
        "cross_fit_audit_checksum": cross_fit_audit["audit_checksum"],
        "fresh_event_ids_sha256": canonical_sha256([_event_id(row) for row in fresh]),
        "replay_event_ids_sha256": canonical_sha256([_event_id(row) for row in replay]),
        "combined_event_ids_sha256": canonical_sha256(
            [
                *({"ledger": "fresh", "event_id": _event_id(row)} for row in fresh),
                *({"ledger": "replay", "event_id": _event_id(row)} for row in replay),
            ]
        ),
        "control_results_sha256": canonical_sha256(control_results),
    }


def _verify_source_evidence_binding(
    metrics: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    fresh_path: Path,
    replay_path: Path,
    cross_fit_path: Path,
) -> None:
    fresh = read_event_ledger(fresh_path)
    replay = read_event_ledger(replay_path)
    cross_fit_audit = read_cross_fit_audit(
        cross_fit_path,
        manifest=manifest,
        source_event_path=fresh_path,
        source_events=fresh,
    )
    expected_cross_fit_summary = {
        "artifact": artifact_descriptor(cross_fit_path),
        "audit_checksum": cross_fit_audit["audit_checksum"],
        "registered_unit_count": cross_fit_audit["registered_unit_count"],
        "checks": dict(cross_fit_audit["checks"]),
        "passed": cross_fit_audit["passed"],
    }
    if metrics.get("cross_fit_audit") != expected_cross_fit_summary:
        raise ManifestDriftError("source metrics/cross-fit audit binding drifted")
    control_results = metrics.get("control_results")
    if not isinstance(control_results, Mapping):
        raise DataGateError("source metrics lack control_results evidence")
    comparator = control_results.get("capacity_matched_independent_posterior")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("cross_fit_schedule_checks") != cross_fit_audit["checks"]
    ):
        raise ManifestDriftError("source comparator/cross-fit audit binding drifted")
    if cross_fit_audit.get("passed") is not True and any(
        comparator.get(field) is True
        for field in ("execution_ok", "scientific_pass", "completed", "passed")
    ):
        raise ManifestDriftError(
            "source comparator claims success over a failed cross-fit audit"
        )
    expected = _source_evidence_binding(
        manifest=manifest,
        fresh_path=fresh_path,
        replay_path=replay_path,
        cross_fit_path=cross_fit_path,
        fresh=fresh,
        replay=replay,
        control_results=control_results,
        cross_fit_audit=cross_fit_audit,
    )
    if metrics.get("source_evidence") != expected:
        raise ManifestDriftError("source controls are not bound to current ledgers")


def _code_bound_source_trainer(
    trainer: Callable[..., Any],
    *,
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> bool:
    candidate = inspect.unwrap(trainer)
    if (
        getattr(candidate, "__module__", None) != "theory.sage_t.t10_2_runtime"
        or getattr(candidate, "__name__", None) != "run_source_trainer"
    ):
        return False
    source_file = inspect.getsourcefile(candidate)
    if source_file is None:
        return False
    expected_path = (repo_root / "theory/sage_t/t10_2_runtime.py").resolve()
    observed_path = Path(source_file).resolve()
    expected_digest = _mapping(manifest.get("code_sha256")).get(
        "theory/sage_t/t10_2_runtime.py"
    )
    return bool(
        observed_path == expected_path
        and isinstance(expected_digest, str)
        and file_sha256(observed_path) == expected_digest
    )


def build_source_gate_report(
    *, manifest: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the registered source controls without interpreting F1 as utility."""

    grammar = _mapping(metrics.get("grammar_oracle"))
    learned = _mapping(metrics.get("learned"))
    completed_controls, controls = _derive_source_control_views(metrics)
    recipe = _mapping(metrics.get("challenger_recipe"))
    recipe_artifact = _mapping(recipe.get("artifact"))
    raw_positive_folds = grammar.get("positive_folds")
    positive_folds = (
        tuple(str(value) for value in raw_positive_folds)
        if isinstance(raw_positive_folds, (list, tuple))
        else ()
    )
    positive_folds_valid = bool(positive_folds) and bool(
        len(set(positive_folds)) == len(positive_folds)
        and set(positive_folds) <= set(SOURCE_GAMES)
    )
    raw_ranks = learned.get("positive_fold_ranks")
    ranks_by_fold: dict[str, int] = {}
    ranks_valid = isinstance(raw_ranks, Mapping)
    if ranks_valid:
        for raw_fold, raw_rank in raw_ranks.items():
            fold = str(raw_fold)
            if isinstance(raw_rank, bool):
                ranks_valid = False
                break
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError):
                ranks_valid = False
                break
            if rank < 1 or rank != raw_rank or fold in ranks_by_fold:
                ranks_valid = False
                break
            ranks_by_fold[fold] = rank
    rank_coverage = bool(
        positive_folds_valid
        and ranks_valid
        and set(ranks_by_fold) == set(positive_folds)
    )
    ranks = tuple(ranks_by_fold.values()) if rank_coverage else ()
    gate = manifest["source_gate"]
    checks = {
        "all_registered_controls_attempted": all(
            result.get("attempted") is True
            for result in _mapping(metrics.get("control_results")).values()
        ),
        "all_registered_controls_execution_ok": all(
            result.get("execution_ok") is True
            for result in _mapping(metrics.get("control_results")).values()
        ),
        "all_registered_controls_complete": all(
            completed_controls.get(name) is True for name in REGISTERED_SOURCE_CONTROLS
        ),
        "grammar_progress_games": int(grammar.get("progress_games", 0))
        >= int(gate["minimum_grammar_progress_games"]),
        "grammar_levels": int(grammar.get("levels", 0))
        >= int(gate["minimum_grammar_levels"]),
        "grammar_zero_errors": int(grammar.get("errors", 1)) == 0,
        "grammar_zero_illegal_actions": int(grammar.get("illegal_actions", 1)) == 0,
        "grammar_zero_game_over": int(grammar.get("game_overs", 1)) == 0,
        "positive_fold_rank_coverage": rank_coverage,
        "positive_fold_top8": bool(ranks)
        and all(rank <= int(gate["maximum_positive_fold_rank"]) for rank in ranks),
        "median_positive_fold_rank": bool(ranks)
        and median(ranks) <= float(gate["maximum_median_positive_fold_rank"]),
        "oracle_level_recovery": float(learned.get("oracle_level_recovery", 0.0))
        >= float(gate["minimum_oracle_level_recovery"]),
        "nonnegative_source_games": int(learned.get("nonnegative_games", 0))
        >= int(gate["minimum_nonnegative_games"]),
        "paired_rate_bootstrap_positive": float(
            learned.get("paired_rate_ci_lower", 0.0)
        )
        > 0.0,
        "no_transport_degrades": float(controls.get("no_transport_degradation", 0.0))
        > 0.0,
        "binding_swap_degrades": float(controls.get("binding_swap_degradation", 0.0))
        > 0.0,
        "transport_oracle_passed": controls.get("transport_oracle_passed") is True,
        "dynamics_oracle_passed": controls.get("dynamics_oracle_passed") is True,
        "goal_oracle_passed": controls.get("goal_oracle_passed") is True,
        "best_executed_sequence_oracle_passed": controls.get(
            "best_executed_sequence_oracle_passed"
        )
        is True,
        "capacity_matched_independent_posterior_passed": controls.get(
            "capacity_matched_independent_posterior_passed"
        )
        is True,
        "option_oracle_passed": controls.get("option_oracle_passed") is True,
        "complete_program_oracle_passed": controls.get("complete_program_oracle_passed")
        is True,
        "common_posterior_passed": learned.get("common_posterior_passed") is True,
        "option_synthesis_passed": learned.get("option_synthesis_passed") is True,
        "identity_probe_increment_closed": _finite_float(
            learned.get("game_seed_probe_accuracy_increment", math.inf), math.inf
        )
        <= float(gate["maximum_game_seed_probe_accuracy_increment"]),
        "zero_illegal_actions": int(learned.get("illegal_actions", 1)) == 0,
        "zero_errors": int(learned.get("errors", 1)) == 0,
        "safety_gate_passed": metrics.get("safety_gate_passed") is True,
        "resource_gate_passed": metrics.get("resource_gate_passed") is True,
        "frozen_challenger_recipe_bound": bool(
            recipe.get("bound") is True
            and recipe.get("path") == CHALLENGER_RECIPE_FILENAME
            and isinstance(recipe_artifact.get("bytes"), int)
            and 0
            < int(recipe_artifact["bytes"])
            <= DEFAULT_RESOURCE_LIMITS.maximum_checkpoint_bytes
            and re.fullmatch(r"[0-9a-f]{64}", str(recipe_artifact.get("sha256", "")))
            and re.fullmatch(r"[0-9a-f]{64}", str(recipe.get("recipe_checksum", "")))
        ),
        "holdout_closed": True,
        "ar25_closed": True,
    }
    grammar_checks = tuple(name for name in checks if name.startswith("grammar_"))
    frame_transport_checks = (
        "no_transport_degrades",
        "binding_swap_degrades",
        "transport_oracle_passed",
    )
    goal_dynamics_checks = (
        "dynamics_oracle_passed",
        "goal_oracle_passed",
        "complete_program_oracle_passed",
    )
    common_posterior_checks = (
        "common_posterior_passed",
        "positive_fold_rank_coverage",
        "positive_fold_top8",
        "median_positive_fold_rank",
    )
    option_checks = (
        "option_synthesis_passed",
        "option_oracle_passed",
    )
    source_grounding_checks = (
        "all_registered_controls_attempted",
        "all_registered_controls_execution_ok",
        "all_registered_controls_complete",
        "capacity_matched_independent_posterior_passed",
        "best_executed_sequence_oracle_passed",
        "oracle_level_recovery",
        "nonnegative_source_games",
        "paired_rate_bootstrap_positive",
        "identity_probe_increment_closed",
        "frozen_challenger_recipe_bound",
    )
    safety_resource_checks = (
        "zero_illegal_actions",
        "zero_errors",
        "safety_gate_passed",
        "resource_gate_passed",
    )
    if not all(checks[name] for name in grammar_checks):
        verdict = "MIXED_SEQUENCE_GRAMMAR_MISS"
    elif not all(checks[name] for name in frame_transport_checks):
        verdict = "FRAME_TRANSPORT_MISS"
    elif not all(checks[name] for name in goal_dynamics_checks):
        verdict = "GOAL_OR_DYNAMICS_MISS"
    elif not all(checks[name] for name in common_posterior_checks):
        verdict = "COMMON_POSTERIOR_MISS"
    elif not all(checks[name] for name in option_checks):
        verdict = "OPTION_SYNTHESIS_MISS"
    elif not all(checks[name] for name in source_grounding_checks):
        verdict = "SOURCE_GROUNDING_MISS"
    elif not all(checks[name] for name in safety_resource_checks):
        verdict = "SAFETY_OR_RESOURCE_MISS"
    else:
        verdict = "PASS_T10_2_SOURCE_GATE"
    passed = verdict == "PASS_T10_2_SOURCE_GATE"
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "source-train",
        "status": "PASS_T10_2_SOURCE_GATE" if passed else "FAIL_T10_2_SOURCE_GATE",
        "verdict": verdict,
        "manifest_checksum": manifest["manifest_checksum"],
        "precollection_aborted_actions": SOURCE_PRECOLLECTION_ABORTED_ACTIONS,
        "maximum_new_actions": SOURCE_MAXIMUM_NEW_ACTIONS,
        "metrics": dict(metrics),
        "registered_controls": {
            name: completed_controls.get(name) is True
            for name in REGISTERED_SOURCE_CONTROLS
        },
        "checks": checks,
        "passed": passed,
        "firewall": {
            "source_validation_opened": passed,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    return signed_payload(payload, checksum_key="report_checksum")


def source_train_phase(
    *,
    metrics: Mapping[str, Any] | None = None,
    trainer: Callable[..., Mapping[str, Any]] | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    compile_report_path: str | Path | None = None,
    replay_report_path: str | Path | None = None,
    fresh_event_path: str | Path | None = None,
    replay_event_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    resource_probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    _test_only_allow_injection: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    if trainer is not None and metrics is not None:
        raise RuntimeUnavailableError(
            "source-train accepts exactly one source evidence producer"
        )
    if metrics is not None and not _test_only_allow_injection:
        raise RuntimeUnavailableError(
            "production source-train forbids injected metrics"
        )
    if (
        trainer is not None
        and not _test_only_allow_injection
        and not _code_bound_source_trainer(
            trainer,
            manifest=manifest,
            repo_root=root,
        )
    ):
        raise RuntimeUnavailableError(
            "production source-train requires the manifest-bound runtime trainer"
        )
    guard_resources(
        repo_root=root,
        limits=limits,
        expensive=True,
        probe=resource_probe,
    )
    destination = Path(output_dir)
    compile_path = Path(compile_report_path or destination / "compile_report.json")
    compile_report = read_checked_json(compile_path)
    if compile_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("compile/manifest binding drifted")
    compile_inputs = {"compile_report": artifact_descriptor(compile_path)}
    if compile_report.get("status") == "DATA_OR_PROVENANCE_INVALID":
        report = build_data_invalid_source_report(
            manifest=manifest,
            reason="compile QA failed before replay or fit",
            compile_report=compile_report,
            inputs=compile_inputs,
        )
        destination.mkdir(parents=True, exist_ok=True)
        write_compact_json(destination / "source_report.json", report)
        enforce_output_artifacts(destination, limits=limits)
        return report
    if not _compile_integrity_passed(compile_report):
        raise GateRefusalError(
            "source training is prohibited until fresh integrity passes"
        )
    replay_path_report = Path(replay_report_path or destination / "replay_report.json")
    replay_report = read_checked_json(replay_path_report)
    if replay_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("replay/manifest binding drifted")
    if replay_report.get("status") != "T10_2_SOURCE_REPLAY_COMPLETE":
        raise GateRefusalError("source training requires the registered replay phase")
    fresh_path = Path(fresh_event_path or destination / "source_events.jsonl")
    replay_path = Path(replay_event_path or destination / "replay_events.jsonl")
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    expected_fresh = _mapping(compile_report.get("inputs")).get("source_events")
    if not isinstance(expected_fresh, Mapping) or dict(
        expected_fresh
    ) != artifact_descriptor(fresh_path):
        raise ManifestDriftError("fresh source ledger drifted after compile")
    expected_cross_fit = _mapping(compile_report.get("inputs")).get("cross_fit_audit")
    if not isinstance(expected_cross_fit, Mapping) or dict(
        expected_cross_fit
    ) != artifact_descriptor(cross_fit_path):
        raise ManifestDriftError("cross-fit audit drifted after compile")
    _verify_artifact_binding(replay_report, replay_path, key="events")
    prefit_inputs = {
        **compile_inputs,
        "replay_report": artifact_descriptor(replay_path_report),
        "fresh_events": artifact_descriptor(fresh_path),
        "replay_events": artifact_descriptor(replay_path),
        "cross_fit_audit": artifact_descriptor(cross_fit_path),
    }
    try:
        prefit_evidence = validate_prefit_evidence(
            manifest=manifest,
            fresh_event_path=fresh_path,
            replay_event_path=replay_path,
            require_pass=False,
        )
    except (DataGateError, FirewallError, ManifestDriftError) as exc:
        report = build_data_invalid_source_report(
            manifest=manifest,
            reason=f"{type(exc).__name__}:{exc}",
            compile_report=compile_report,
            inputs=prefit_inputs,
        )
        destination.mkdir(parents=True, exist_ok=True)
        write_compact_json(destination / "source_report.json", report)
        enforce_output_artifacts(destination, limits=limits)
        return report
    if prefit_evidence.get("combined_qa_status") != "PASS_T10_2_QA":
        report = build_data_invalid_source_report(
            manifest=manifest,
            reason="combined fresh+replay QA failed before fit",
            compile_report=compile_report,
            prefit_evidence=prefit_evidence,
            inputs=prefit_inputs,
        )
        destination.mkdir(parents=True, exist_ok=True)
        write_compact_json(destination / "source_report.json", report)
        enforce_output_artifacts(destination, limits=limits)
        return report
    if trainer is not None:
        observed_metrics = _invoke_with_context(
            trainer,
            context={
                "manifest": manifest,
                "compile_report": compile_report,
                "replay_report": replay_report,
                "output_dir": destination,
            },
            positional_fallback=(manifest, compile_report, replay_report),
        )
    elif metrics is not None:
        observed_metrics = metrics
    else:
        raise RuntimeUnavailableError(
            "source-train requires the manifest-bound runtime trainer"
        )
    if not isinstance(observed_metrics, Mapping):
        raise DataGateError("source trainer did not return a metrics mapping")
    observed_metrics = dict(observed_metrics)
    _derive_source_control_views(observed_metrics)
    _verify_source_evidence_binding(
        observed_metrics,
        manifest=manifest,
        fresh_path=fresh_path,
        replay_path=replay_path,
        cross_fit_path=cross_fit_path,
    )
    observed_metrics["challenger_recipe"] = _verified_challenger_recipe_binding(
        observed_metrics,
        output_dir=destination,
        manifest=manifest,
        limits=limits,
    )
    guard_resources(
        repo_root=root,
        limits=limits,
        expensive=True,
        probe=resource_probe,
    )
    report = build_source_gate_report(
        manifest=manifest,
        metrics=observed_metrics,
    )
    report["inputs"] = {
        **prefit_inputs,
        "prefit_evidence": prefit_evidence,
    }
    report = signed_payload(report, checksum_key="report_checksum")
    destination.mkdir(parents=True, exist_ok=True)
    write_compact_json(destination / "source_report.json", report)
    enforce_output_artifacts(destination, limits=limits)
    return report


def require_source_gate(
    *,
    source_report_path: str | Path,
    manifest: Mapping[str, Any],
    output_dir: str | Path | None = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    """Reconstruct source authorization before validation can see a game."""

    source_path = Path(source_report_path)
    report = read_checked_json(source_path)
    if report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("source report/manifest binding drifted")
    report = _reconstruct_source_report(
        manifest=manifest,
        source_report=report,
        destination=Path(output_dir or source_path.parent),
        limits=limits,
    )
    checks = report.get("checks")
    registered_controls = report.get("registered_controls")
    if (
        report.get("status") != "PASS_T10_2_SOURCE_GATE"
        or report.get("verdict") != "PASS_T10_2_SOURCE_GATE"
        or report.get("passed") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(registered_controls, Mapping)
        or set(registered_controls) != set(REGISTERED_SOURCE_CONTROLS)
        or not all(value is True for value in registered_controls.values())
    ):
        raise GateRefusalError("validation refused: source gate did not pass")
    firewall = report.get("firewall", {})
    if firewall.get("source_validation_opened") is not True or any(
        bool(firewall.get(key))
        for key in ("ar25_opened", "holdout_opened", "production_authority")
    ):
        raise GateRefusalError("validation refused: source firewall is invalid")
    return report


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result < 0 or result != value:
        return None
    return result


def _latency_samples(value: Any) -> tuple[list[float], bool]:
    if not isinstance(value, (list, tuple)):
        return [], False
    samples: list[float] = []
    for raw in value:
        sample = _finite_float(raw, math.inf)
        if sample < 0.0 or not math.isfinite(sample):
            return [], False
        samples.append(sample)
    return samples, True


_VALIDATION_RUN_FIELDS = frozenset(
    {
        "game_id",
        "seed",
        "baseline",
        "t10_2",
        "registered_resets_per_controller",
        "registered_max_actions_per_reset",
        "controller_order",
        "counterbalanced",
        "posterior_reset",
        "learning_between_controllers",
        "wall_seconds",
    }
)
_VALIDATION_ARM_REQUIRED_FIELDS = frozenset(
    {
        "levels",
        "legal_actions",
        "game_overs",
        "illegal_actions",
        "errors",
        "planned_actions",
        "completed_actions",
        "unregistered_stops",
        "reset_summaries",
        "decision_latency_ms",
        "observation_latency_ms",
    }
)
_VALIDATION_ARM_OPTIONAL_FIELDS = frozenset(
    {
        "behavior_projection",
        "retained_observations",
        "deduplicated_immediate_noops",
        "online_observations",
    }
)
_VALIDATION_RESET_FIELDS = frozenset(
    {"reset_index", "planned_actions", "completed_actions", "stop_reason"}
)
_VALIDATION_ERROR_STOP_REASONS = frozenset(
    {"decision_error", "step_error", "observation_error"}
)


def _require_closed_validation_fields(
    value: Mapping[str, Any], *, allowed: frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != set(allowed):
        missing = sorted(allowed - observed)
        unknown = sorted(observed - allowed, key=str)
        raise DataGateError(
            f"{label} schema drifted; missing={missing}, unknown={unknown}"
        )


def _validate_validation_run_summary(row: Mapping[str, Any]) -> None:
    """Reject non-compact validation evidence before aggregation or persistence."""

    transferable = {
        key: value for key, value in row.items() if key not in {"game_id", "seed"}
    }
    hits = forbidden_transfer_payload_hits(transferable, root="validation_summary")
    if hits:
        raise FirewallError(
            "forbidden evidence in validation summary: " + ", ".join(hits)
        )
    _require_closed_validation_fields(
        row,
        allowed=_VALIDATION_RUN_FIELDS,
        label="validation pair",
    )
    resets = _strict_nonnegative_int(row.get("registered_resets_per_controller"))
    actions_per_reset = _strict_nonnegative_int(
        row.get("registered_max_actions_per_reset")
    )
    if resets != VALIDATION_RESETS_PER_GAME_SEED:
        raise DataGateError("validation pair reset count drifted")
    if actions_per_reset != VALIDATION_ACTIONS_PER_RESET:
        raise DataGateError("validation pair action budget drifted")
    if not isinstance(row.get("controller_order"), (list, tuple)) or any(
        not isinstance(item, str) for item in row["controller_order"]
    ):
        raise DataGateError("validation controller order is not a string sequence")
    for field in (
        "counterbalanced",
        "posterior_reset",
        "learning_between_controllers",
    ):
        if not isinstance(row.get(field), bool):
            raise DataGateError(f"validation pair lacks boolean {field}")
    wall_seconds = row.get("wall_seconds")
    parsed_wall_seconds = _finite_float(wall_seconds, math.inf)
    if (
        isinstance(wall_seconds, bool)
        or not math.isfinite(parsed_wall_seconds)
        or parsed_wall_seconds < 0.0
    ):
        raise DataGateError("validation pair wall time is not finite and nonnegative")

    allowed_arm_fields = (
        _VALIDATION_ARM_REQUIRED_FIELDS | _VALIDATION_ARM_OPTIONAL_FIELDS
    )
    for controller_name in ("baseline", "t10_2"):
        controller = row.get(controller_name)
        if not isinstance(controller, Mapping):
            raise DataGateError(
                f"validation {controller_name} summary is not a mapping"
            )
        observed_arm_fields = set(controller)
        missing_arm_fields = _VALIDATION_ARM_REQUIRED_FIELDS - observed_arm_fields
        unknown_arm_fields = observed_arm_fields - allowed_arm_fields
        if missing_arm_fields or unknown_arm_fields:
            raise DataGateError(
                f"validation {controller_name} schema drifted; "
                f"missing={sorted(missing_arm_fields)}, "
                f"unknown={sorted(unknown_arm_fields, key=str)}"
            )
        counts: dict[str, int] = {}
        for field in (
            "levels",
            "legal_actions",
            "game_overs",
            "illegal_actions",
            "errors",
            "planned_actions",
            "completed_actions",
            "unregistered_stops",
        ):
            parsed = _strict_nonnegative_int(controller.get(field))
            if parsed is None:
                raise DataGateError(f"validation {controller_name} has invalid {field}")
            counts[field] = parsed
        for field in (
            "retained_observations",
            "deduplicated_immediate_noops",
            "online_observations",
        ):
            if (
                field in controller
                and _strict_nonnegative_int(controller[field]) is None
            ):
                raise DataGateError(f"validation {controller_name} has invalid {field}")
        if "behavior_projection" in controller and (
            not isinstance(controller["behavior_projection"], str)
            or not controller["behavior_projection"]
            or len(controller["behavior_projection"]) > 256
        ):
            raise DataGateError(
                f"validation {controller_name} behavior projection is invalid"
            )
        reset_summaries = controller.get("reset_summaries")
        if (
            not isinstance(reset_summaries, (list, tuple))
            or len(reset_summaries) != resets
        ):
            raise DataGateError(
                f"validation {controller_name} lacks exactly {resets} reset summaries"
            )
        planned_total = 0
        completed_total = 0
        unregistered_total = 0
        game_over_total = 0
        error_total = 0
        illegal_total = 0
        for reset_index, reset_summary in enumerate(reset_summaries):
            if not isinstance(reset_summary, Mapping):
                raise DataGateError(
                    f"validation {controller_name} reset summary is not a mapping"
                )
            _require_closed_validation_fields(
                reset_summary,
                allowed=_VALIDATION_RESET_FIELDS,
                label=f"validation {controller_name} reset",
            )
            observed_reset_index = _strict_nonnegative_int(
                reset_summary.get("reset_index")
            )
            planned = _strict_nonnegative_int(reset_summary.get("planned_actions"))
            completed = _strict_nonnegative_int(reset_summary.get("completed_actions"))
            reason = reset_summary.get("stop_reason")
            if observed_reset_index != reset_index:
                raise DataGateError(
                    f"validation {controller_name} reset ordering drifted"
                )
            if planned is None or completed is None or completed > planned:
                raise DataGateError(
                    f"validation {controller_name} reset action counts are invalid"
                )
            if reason not in VALIDATION_STOP_REASONS:
                raise DataGateError(
                    f"validation {controller_name} reset stop reason is unregistered"
                )
            if reason in VALIDATION_EXEMPT_STOP_REASONS and planned != completed:
                raise DataGateError(
                    f"validation {controller_name} exempt stop changed its denominator"
                )
            if reason == "budget_exhausted" and (
                planned != actions_per_reset or completed != actions_per_reset
            ):
                raise DataGateError(
                    f"validation {controller_name} budget exhaustion is inconsistent"
                )
            if (
                reason in VALIDATION_UNREGISTERED_STOP_REASONS
                and planned != actions_per_reset
            ):
                raise DataGateError(
                    f"validation {controller_name} early stop shrank its denominator"
                )
            planned_total += planned
            completed_total += completed
            unregistered_total += int(reason in VALIDATION_UNREGISTERED_STOP_REASONS)
            game_over_total += int(reason == "game_over")
            error_total += int(reason in _VALIDATION_ERROR_STOP_REASONS)
            illegal_total += int(reason == "illegal_action")
        if counts["planned_actions"] != planned_total:
            raise DataGateError(
                f"validation {controller_name} planned-action total drifted"
            )
        if counts["completed_actions"] != completed_total:
            raise DataGateError(
                f"validation {controller_name} completed-action total drifted"
            )
        if counts["legal_actions"] != completed_total:
            raise DataGateError(
                f"validation {controller_name} legal-action total drifted"
            )
        if counts["unregistered_stops"] != unregistered_total:
            raise DataGateError(
                f"validation {controller_name} unregistered-stop total drifted"
            )
        if counts["game_overs"] != game_over_total:
            raise DataGateError(f"validation {controller_name} game-over total drifted")
        if counts["errors"] != error_total:
            raise DataGateError(f"validation {controller_name} error total drifted")
        if counts["illegal_actions"] != illegal_total:
            raise DataGateError(
                f"validation {controller_name} illegal-action total drifted"
            )
        for latency_field in ("decision_latency_ms", "observation_latency_ms"):
            samples, valid = _latency_samples(controller.get(latency_field))
            if not valid or len(samples) > counts["planned_actions"]:
                raise DataGateError(
                    f"validation {controller_name} has invalid {latency_field}"
                )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_lower(
    values: Sequence[float], *, samples: int = 10_000, seed: int = 10_202
) -> float:
    """Deterministic paired 95% lower bound over game-seed units."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    generator = random.Random(seed)
    means = []
    count = len(finite)
    for _ in range(samples):
        means.append(sum(finite[generator.randrange(count)] for _ in finite) / count)
    return _percentile(means, 0.025)


def aggregate_validation_runs(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate paired T10.1/T10.2 summaries without retaining raw frames."""

    rate_deltas: list[float] = []
    per_game_levels: Counter[str] = Counter()
    candidate_levels = 0
    baseline_levels = 0
    controller_totals = {
        "baseline": {"legal": 0, "planned": 0, "completed": 0, "unregistered": 0},
        "candidate": {"legal": 0, "planned": 0, "completed": 0, "unregistered": 0},
    }
    candidate_game_overs = 0
    baseline_game_overs = 0
    illegal_actions = 0
    errors = 0
    decision_latencies: list[float] = []
    observation_latencies: list[float] = []
    wall_seconds = 0.0
    seen_pairs: set[tuple[str, int]] = set()
    pairing_flags: list[bool] = []
    budget_configuration_flags: list[bool] = []
    action_cap_flags: list[bool] = []
    latency_schema_flags: list[bool] = []

    for pair_index, row in enumerate(runs):
        _validate_validation_run_summary(row)
        game = str(row.get("game_id", ""))
        try:
            seed = int(row.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise DataGateError("invalid validation seed") from exc
        if game not in VALIDATION_GAMES or seed not in VALIDATION_SEEDS:
            raise FirewallError(f"unregistered validation result: {game}/{seed}")
        pair = (game, seed)
        if pair in seen_pairs:
            raise DataGateError(f"duplicate validation pair: {game}/{seed}")
        seen_pairs.add(pair)
        baseline = _mapping(row.get("baseline"))
        candidate = _mapping(row.get("t10_2"))
        resets = _strict_nonnegative_int(row.get("registered_resets_per_controller"))
        actions_per_reset = _strict_nonnegative_int(
            row.get("registered_max_actions_per_reset")
        )
        budget_configuration_flags.append(
            resets == VALIDATION_RESETS_PER_GAME_SEED
            and actions_per_reset == VALIDATION_ACTIONS_PER_RESET
        )
        pair_action_cap = (
            resets * actions_per_reset
            if resets is not None and actions_per_reset is not None
            else -1
        )

        parsed: dict[str, dict[str, int]] = {}
        for controller_name, controller in (
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            values = {
                field: _strict_nonnegative_int(controller.get(field))
                for field in (
                    "levels",
                    "legal_actions",
                    "game_overs",
                    "illegal_actions",
                    "errors",
                    "planned_actions",
                    "completed_actions",
                    "unregistered_stops",
                )
            }
            counts_valid = all(value is not None for value in values.values())
            counts = {field: int(value or 0) for field, value in values.items()}
            parsed[controller_name] = counts
            action_cap_flags.append(
                bool(
                    counts_valid
                    and pair_action_cap
                    == VALIDATION_RESETS_PER_GAME_SEED * VALIDATION_ACTIONS_PER_RESET
                    and counts["legal_actions"] <= pair_action_cap
                    and counts["planned_actions"] <= pair_action_cap
                    and counts["completed_actions"] <= pair_action_cap
                    and counts["legal_actions"] <= counts["completed_actions"]
                    and counts["completed_actions"] <= counts["planned_actions"]
                    and counts["game_overs"] <= VALIDATION_RESETS_PER_GAME_SEED
                )
            )
            totals = controller_totals[controller_name]
            totals["legal"] += counts["legal_actions"]
            totals["planned"] += counts["planned_actions"]
            totals["completed"] += counts["completed_actions"]
            totals["unregistered"] += counts["unregistered_stops"]

        baseline_counts = parsed["baseline"]
        candidate_counts = parsed["candidate"]
        baseline_pair_levels = baseline_counts["levels"]
        candidate_pair_levels = candidate_counts["levels"]
        baseline_pair_legal = baseline_counts["legal_actions"]
        candidate_pair_legal = candidate_counts["legal_actions"]
        baseline_levels += baseline_pair_levels
        candidate_levels += candidate_pair_levels
        level_delta = candidate_pair_levels - baseline_pair_levels
        per_game_levels[game] += level_delta
        baseline_rate = (
            1000.0 * baseline_pair_levels / baseline_pair_legal
            if baseline_pair_legal
            else 0.0
        )
        candidate_rate = (
            1000.0 * candidate_pair_levels / candidate_pair_legal
            if candidate_pair_legal
            else 0.0
        )
        rate_deltas.append(candidate_rate - baseline_rate)
        candidate_game_overs += candidate_counts["game_overs"]
        baseline_game_overs += baseline_counts["game_overs"]
        illegal_actions += (
            candidate_counts["illegal_actions"] + baseline_counts["illegal_actions"]
        )
        errors += candidate_counts["errors"] + baseline_counts["errors"]
        pair_decision_latencies, decision_valid = _latency_samples(
            candidate.get("decision_latency_ms")
        )
        pair_observation_latencies, observation_valid = _latency_samples(
            candidate.get("observation_latency_ms")
        )
        decision_latencies.extend(pair_decision_latencies)
        observation_latencies.extend(pair_observation_latencies)
        latency_schema_flags.append(decision_valid and observation_valid)
        wall_seconds += _finite_float(row.get("wall_seconds", math.inf), math.inf)
        expected_order = (
            ("t10_1", "t10_2") if pair_index % 2 == 0 else ("t10_2", "t10_1")
        )
        pairing_flags.append(
            row.get("counterbalanced") is True
            and row.get("posterior_reset") is True
            and row.get("learning_between_controllers") is False
            and tuple(row.get("controller_order", ())) == expected_order
        )

    expected_pairs = {
        (game, seed) for game in VALIDATION_GAMES for seed in VALIDATION_SEEDS
    }
    scheduled_episodes_per_controller = len(runs) * VALIDATION_RESETS_PER_GAME_SEED
    maximum_actions_per_controller = (
        scheduled_episodes_per_controller * VALIDATION_ACTIONS_PER_RESET
    )
    candidate_totals = controller_totals["candidate"]
    baseline_totals = controller_totals["baseline"]
    planned_actions = candidate_totals["planned"] + baseline_totals["planned"]
    completed_actions = candidate_totals["completed"] + baseline_totals["completed"]
    candidate_completed_fraction = (
        candidate_totals["completed"] / candidate_totals["planned"]
        if candidate_totals["planned"]
        else 1.0
    )
    baseline_completed_fraction = (
        baseline_totals["completed"] / baseline_totals["planned"]
        if baseline_totals["planned"]
        else 1.0
    )
    within_action_caps = bool(
        action_cap_flags
        and all(action_cap_flags)
        and all(
            totals[field] <= VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
            for totals in controller_totals.values()
            for field in ("legal", "planned", "completed")
        )
    )
    latency_samples_complete = bool(
        latency_schema_flags
        and all(latency_schema_flags)
        and candidate_totals["completed"] > 0
        and len(decision_latencies) == candidate_totals["completed"]
        and len(observation_latencies) == candidate_totals["completed"]
    )
    return {
        "all_pairs_executed": seen_pairs == expected_pairs,
        "counterbalanced_and_reset": bool(pairing_flags) and all(pairing_flags),
        "total_level_advantage": candidate_levels - baseline_levels,
        "nonnegative_games": sum(value >= 0 for value in per_game_levels.values()),
        "per_game_level_advantage": dict(sorted(per_game_levels.items())),
        "paired_rate_ci_lower": paired_bootstrap_lower(rate_deltas),
        "illegal_actions": illegal_actions,
        "errors": errors,
        "unregistered_stops": (
            candidate_totals["unregistered"] + baseline_totals["unregistered"]
        ),
        "game_over_rate_delta": (
            candidate_game_overs / scheduled_episodes_per_controller
            - baseline_game_overs / scheduled_episodes_per_controller
            if scheduled_episodes_per_controller
            else 1.0
        ),
        "scheduled_episodes_per_controller": scheduled_episodes_per_controller,
        "candidate_game_overs": candidate_game_overs,
        "baseline_game_overs": baseline_game_overs,
        "budget_configuration_exact": bool(
            budget_configuration_flags
            and all(budget_configuration_flags)
            and maximum_actions_per_controller
            == VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
        ),
        "within_action_caps": within_action_caps,
        "maximum_actions_per_controller": maximum_actions_per_controller,
        "completed_budget_fraction": min(
            candidate_completed_fraction, baseline_completed_fraction
        ),
        "candidate_completed_budget_fraction": candidate_completed_fraction,
        "baseline_completed_budget_fraction": baseline_completed_fraction,
        "combined_completed_budget_fraction": (
            completed_actions / planned_actions if planned_actions else 1.0
        ),
        "decision_latency_sample_count": len(decision_latencies),
        "observation_latency_sample_count": len(observation_latencies),
        "latency_samples_complete": latency_samples_complete,
        "decision_p95_ms": _percentile(decision_latencies, 0.95),
        "decision_p99_ms": _percentile(decision_latencies, 0.99),
        "observation_p95_ms": _percentile(observation_latencies, 0.95),
        "observation_p99_ms": _percentile(observation_latencies, 0.99),
        "wall_seconds": wall_seconds,
        "candidate_actions": candidate_totals["legal"],
        "baseline_actions": baseline_totals["legal"],
    }


def build_validation_report(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    gate = manifest["validation_gate"]
    checks = {
        "all_pairs_executed": metrics.get("all_pairs_executed") is True,
        "counterbalanced_and_reset": metrics.get("counterbalanced_and_reset") is True,
        "total_level_advantage": int(metrics.get("total_level_advantage", 0))
        >= int(gate["minimum_total_level_advantage"]),
        "nonnegative_games": int(metrics.get("nonnegative_games", 0))
        >= int(gate["minimum_nonnegative_games"]),
        "paired_bootstrap_positive": _finite_float(
            metrics.get("paired_rate_ci_lower", 0.0)
        )
        > 0.0,
        "zero_illegal_actions": int(metrics.get("illegal_actions", 1)) == 0,
        "zero_errors": int(metrics.get("errors", 1)) == 0,
        "zero_unregistered_stops": int(metrics.get("unregistered_stops", 1)) == 0,
        "game_over_not_worse": _finite_float(metrics.get("game_over_rate_delta", 1.0))
        <= 0.0,
        "exact_validation_budget": bool(
            metrics.get("budget_configuration_exact") is True
            and metrics.get("within_action_caps") is True
            and int(metrics.get("maximum_actions_per_controller", -1))
            == VALIDATION_MAXIMUM_ACTIONS_PER_CONTROLLER
        ),
        "planned_budget_completion": _finite_float(
            metrics.get("completed_budget_fraction", 0.0)
        )
        >= float(gate["minimum_completed_budget_fraction"]),
        "latency_samples_present": bool(
            metrics.get("latency_samples_complete") is True
            and int(metrics.get("decision_latency_sample_count", 0)) > 0
            and int(metrics.get("observation_latency_sample_count", 0)) > 0
        ),
        "decision_p95": _finite_float(
            metrics.get("decision_p95_ms", math.inf), math.inf
        )
        <= float(gate["maximum_decision_p95_ms"]),
        "decision_p99": _finite_float(
            metrics.get("decision_p99_ms", math.inf), math.inf
        )
        <= float(gate["maximum_decision_p99_ms"]),
        "observation_p95": _finite_float(
            metrics.get("observation_p95_ms", math.inf), math.inf
        )
        <= float(gate["maximum_observation_p95_ms"]),
        "observation_p99": _finite_float(
            metrics.get("observation_p99_ms", math.inf), math.inf
        )
        <= float(gate["maximum_observation_p99_ms"]),
        "wall_time": _finite_float(metrics.get("wall_seconds", math.inf), math.inf)
        <= float(gate["maximum_wall_seconds"]),
        "holdout_closed": True,
        "ar25_closed": True,
        "production_authority_closed": True,
    }
    passed = all(checks.values())
    safety_names = (
        "zero_illegal_actions",
        "zero_errors",
        "zero_unregistered_stops",
        "game_over_not_worse",
        "exact_validation_budget",
        "planned_budget_completion",
        "latency_samples_present",
        "decision_p95",
        "decision_p99",
        "observation_p95",
        "observation_p99",
        "wall_time",
    )
    transfer_names = (
        "all_pairs_executed",
        "counterbalanced_and_reset",
        "total_level_advantage",
        "nonnegative_games",
        "paired_bootstrap_positive",
    )
    if passed:
        verdict = "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED"
    elif not all(checks[name] for name in transfer_names):
        verdict = "SOURCE_VALIDATION_TRANSFER_MISS"
    elif not all(checks[name] for name in safety_names):
        verdict = "SAFETY_OR_RESOURCE_MISS"
    else:  # pragma: no cover - every registered check belongs to one group.
        raise AssertionError("unclassified T10.2 validation gate failure")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "validate",
        "status": "PASS_T10_2_VALIDATION" if passed else "FAIL_T10_2_VALIDATION",
        "verdict": verdict,
        "manifest_checksum": manifest["manifest_checksum"],
        "source_report_checksum": source_report["report_checksum"],
        "metrics": dict(metrics),
        "checks": checks,
        "passed": passed,
        "firewall": {
            "source_validation_opened": True,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    return signed_payload(payload, checksum_key="report_checksum")


def _validation_timing_code_binding(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    registry = manifest.get("code_sha256")
    if not isinstance(registry, Mapping):
        raise ManifestDriftError("manifest lacks validation timing code registry")
    binding: dict[str, str] = {}
    for path in VALIDATION_TIMING_CODE_PATHS:
        digest = registry.get(path)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ManifestDriftError(
                f"manifest lacks validation timing code binding: {path}"
            )
        binding[path] = digest
    return binding


def _build_validation_timing_proof(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
    runs_path: Path,
    runs: Sequence[Mapping[str, Any]],
    monotonic_started: float,
    monotonic_finished: float,
    reported_pair_wall_seconds: float,
) -> dict[str, Any]:
    elapsed = monotonic_finished - monotonic_started
    for label, value in (
        ("monotonic_started", monotonic_started),
        ("monotonic_finished", monotonic_finished),
        ("monotonic_elapsed_seconds", elapsed),
        ("reported_pair_wall_seconds", reported_pair_wall_seconds),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise DataGateError(f"validation timing proof has invalid {label}")
    if monotonic_finished < monotonic_started:
        raise DataGateError("validation monotonic clock moved backwards")
    payload: dict[str, Any] = {
        "format_version": VALIDATION_TIMING_PROOF_FORMAT_VERSION,
        "producer": "validate_phase:external_monotonic_clock_v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "source_report": artifact_descriptor(source_path),
        "source_report_checksum": source_report["report_checksum"],
        "validation_runs": artifact_descriptor(runs_path),
        "validation_run_count": len(runs),
        "validation_pairs_sha256": canonical_sha256(
            [[str(row.get("game_id", "")), int(row.get("seed", -1))] for row in runs]
        ),
        "reported_pair_wall_seconds": reported_pair_wall_seconds,
        "monotonic_started": monotonic_started,
        "monotonic_finished": monotonic_finished,
        "monotonic_elapsed_seconds": elapsed,
        "code_sha256": _validation_timing_code_binding(manifest),
        "validation_report_linked": False,
        "cycle_free": True,
    }
    return signed_payload(payload, checksum_key="timing_proof_checksum")


def _read_validation_timing_proof(
    *,
    path: Path,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
    runs_path: Path,
    runs: Sequence[Mapping[str, Any]],
    reported_pair_wall_seconds: float,
) -> dict[str, Any]:
    proof = read_checked_json(path, checksum_key="timing_proof_checksum")
    expected_keys = {
        "format_version",
        "producer",
        "manifest_checksum",
        "source_report",
        "source_report_checksum",
        "validation_runs",
        "validation_run_count",
        "validation_pairs_sha256",
        "reported_pair_wall_seconds",
        "monotonic_started",
        "monotonic_finished",
        "monotonic_elapsed_seconds",
        "code_sha256",
        "validation_report_linked",
        "cycle_free",
        "timing_proof_checksum",
    }
    if set(proof) != expected_keys:
        raise ManifestDriftError("validation timing proof schema drifted")
    expected_static = {
        "format_version": VALIDATION_TIMING_PROOF_FORMAT_VERSION,
        "producer": "validate_phase:external_monotonic_clock_v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "source_report": artifact_descriptor(source_path),
        "source_report_checksum": source_report["report_checksum"],
        "validation_runs": artifact_descriptor(runs_path),
        "validation_run_count": len(runs),
        "validation_pairs_sha256": canonical_sha256(
            [[str(row.get("game_id", "")), int(row.get("seed", -1))] for row in runs]
        ),
        "reported_pair_wall_seconds": reported_pair_wall_seconds,
        "code_sha256": _validation_timing_code_binding(manifest),
        "validation_report_linked": False,
        "cycle_free": True,
    }
    for key, expected in expected_static.items():
        if canonical_json(proof.get(key)) != canonical_json(expected):
            raise ManifestDriftError(f"validation timing proof {key} drifted")
    timing_values: dict[str, float] = {}
    for field in (
        "monotonic_started",
        "monotonic_finished",
        "monotonic_elapsed_seconds",
    ):
        value = proof.get(field)
        if isinstance(value, bool):
            raise ManifestDriftError(f"validation timing proof {field} is invalid")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ManifestDriftError(
                f"validation timing proof {field} is invalid"
            ) from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ManifestDriftError(f"validation timing proof {field} is invalid")
        timing_values[field] = parsed
    expected_elapsed = (
        timing_values["monotonic_finished"] - timing_values["monotonic_started"]
    )
    if expected_elapsed < 0.0 or not math.isclose(
        timing_values["monotonic_elapsed_seconds"],
        expected_elapsed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ManifestDriftError("validation timing proof elapsed duration drifted")
    return proof


def validate_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    env_factory: Callable[..., Any] | None = None,
    games: Sequence[str] | None = None,
    resource_probe: Callable[[str | Path | None], ResourceSnapshot] = resource_snapshot,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    clock: Callable[[], float] = time.perf_counter,
    _test_only_allow_clock: bool = False,
) -> dict[str, Any]:
    if clock is not time.perf_counter and not _test_only_allow_clock:
        raise RuntimeUnavailableError(
            "production validation requires the code-bound monotonic clock"
        )
    previous_clock: float | None = None

    def sample_clock(label: str) -> float:
        nonlocal previous_clock
        try:
            observed = float(clock())
        except (TypeError, ValueError) as exc:
            raise DataGateError(f"validation monotonic {label} is invalid") from exc
        if not math.isfinite(observed) or (
            previous_clock is not None and observed < previous_clock
        ):
            raise DataGateError(f"validation monotonic {label} regressed")
        previous_clock = observed
        return observed

    validation_started = sample_clock("start")
    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    destination = Path(output_dir)

    # This signed source authorization is intentionally checked before any
    # validation game, resource-heavy setup, or environment factory.
    source_path = Path(source_report_path or destination / "source_report.json")
    source_report = require_source_gate(
        source_report_path=source_path,
        manifest=manifest,
        output_dir=destination,
        limits=limits,
    )
    selected_games = tuple(games or VALIDATION_GAMES)
    for game_id in selected_games:
        enforce_environment_firewall(
            phase="validate", game_id=game_id, source_gate_passed=True
        )
    if selected_games != VALIDATION_GAMES:
        raise FirewallError("validation must use the exact frozen game allowlist")
    guard_resources(
        repo_root=root,
        limits=limits,
        expensive=True,
        probe=resource_probe,
    )
    factory = env_factory or _default_env_factory
    runs: list[dict[str, Any]] = []
    for pair_index, (game_id, seed) in enumerate(
        (game, seed) for game in selected_games for seed in VALIDATION_SEEDS
    ):
        enforce_environment_firewall(
            phase="validate", game_id=game_id, source_gate_passed=True
        )
        environment = _call_env_factory(
            factory,
            game_id=game_id,
            seed=seed,
            phase="validate",
            split="paired_validation",
        )
        try:
            raw_runs = _environment_rows(
                environment,
                context={
                    "game_id": game_id,
                    "seed": seed,
                    "split": "paired_validation",
                    "resets": VALIDATION_RESETS_PER_GAME_SEED,
                    "action_budget": VALIDATION_ACTIONS_PER_RESET,
                    "controller_order": (
                        ("t10_1", "t10_2")
                        if pair_index % 2 == 0
                        else ("t10_2", "t10_1")
                    ),
                    "posterior_reset": True,
                    "learning_enabled": False,
                },
            )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        if len(raw_runs) != 1 or not isinstance(raw_runs[0], Mapping):
            raise DataGateError(
                "validation adapter must return one paired summary per game-seed"
            )
        run = dict(raw_runs[0])
        for field, expected in (
            ("registered_resets_per_controller", VALIDATION_RESETS_PER_GAME_SEED),
            ("registered_max_actions_per_reset", VALIDATION_ACTIONS_PER_RESET),
        ):
            if field in run and _strict_nonnegative_int(run[field]) != expected:
                raise DataGateError(f"validation result drifted from {field}")
            run[field] = expected
        run.setdefault("game_id", game_id)
        run.setdefault("seed", seed)
        if str(run["game_id"]) != game_id or int(run["seed"]) != seed:
            raise DataGateError("validation result escaped its registered pair")
        _validate_validation_run_summary(run)
        runs.append(run)
        guard_resources(
            repo_root=root,
            limits=limits,
            expensive=True,
            probe=resource_probe,
        )
    metrics = aggregate_validation_runs(runs)
    metrics["reported_pair_wall_seconds"] = metrics["wall_seconds"]
    destination.mkdir(parents=True, exist_ok=True)
    runs_path = destination / "validation_runs.jsonl"
    _atomic_write_lines(
        runs_path,
        (canonical_json(row) + "\n" for row in runs),
    )
    timing_path = destination / VALIDATION_TIMING_PROOF_FILENAME
    report_path = destination / "validation_report.json"

    def persist_validation_evidence(finished: float) -> dict[str, Any]:
        timing_proof = _build_validation_timing_proof(
            manifest=manifest,
            source_report=source_report,
            source_path=source_path,
            runs_path=runs_path,
            runs=runs,
            monotonic_started=validation_started,
            monotonic_finished=finished,
            reported_pair_wall_seconds=float(metrics["reported_pair_wall_seconds"]),
        )
        write_compact_json(timing_path, timing_proof)
        enforce_artifact_limit(timing_path, kind="derived", limits=limits)
        persisted_metrics = dict(metrics)
        persisted_metrics["wall_seconds"] = timing_proof["monotonic_elapsed_seconds"]
        persisted_report = build_validation_report(
            manifest=manifest,
            source_report=source_report,
            metrics=persisted_metrics,
        )
        persisted_report["inputs"] = {
            "source_report": artifact_descriptor(source_path),
            "validation_runs": artifact_descriptor(runs_path),
            "validation_timing_proof": artifact_descriptor(timing_path),
        }
        persisted_report = signed_payload(
            persisted_report,
            checksum_key="report_checksum",
        )
        write_compact_json(report_path, persisted_report)
        enforce_output_artifacts(destination, limits=limits)
        return persisted_report

    # First materialize a complete evidence set, then sample the phase clock.
    # The second deterministic pass records an interval that includes ledger,
    # proof, report, fsync and output-limit enforcement from the first pass.
    preliminary_finished = sample_clock("preliminary finish")
    persist_validation_evidence(preliminary_finished)
    validation_finished = sample_clock("persistence finish")
    report = persist_validation_evidence(validation_finished)
    post_persistence_elapsed = (
        sample_clock("post-persistence recheck") - validation_started
    )
    if post_persistence_elapsed > VALIDATION_MAXIMUM_WALL_SECONDS:
        raise ResourceGateError("validation exceeded 21,600 seconds after persistence")
    return report


def _require_report_input(
    report: Mapping[str, Any],
    *,
    key: str,
    path: Path,
) -> None:
    descriptor = _mapping(report.get("inputs")).get(key)
    if not isinstance(descriptor, Mapping) or dict(descriptor) != artifact_descriptor(
        path
    ):
        raise ManifestDriftError(f"report input binding drifted: {key}")


def _compare_reconstructed_fields(
    stored: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
    *,
    fields: Sequence[str],
    label: str,
) -> None:
    for field in fields:
        try:
            stored_value = canonical_json(stored.get(field))
            reconstructed_value = canonical_json(reconstructed.get(field))
        except (TypeError, ValueError) as exc:
            raise ManifestDriftError(f"{label} contains noncanonical {field}") from exc
        if stored_value != reconstructed_value:
            raise ManifestDriftError(f"{label} reconstructed {field} drifted")


def _reconstruct_source_report(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    destination: Path,
    limits: ResourceLimits,
) -> dict[str, Any]:
    if source_report.get("status") == "DATA_OR_PROVENANCE_INVALID":
        compile_path = destination / "compile_report.json"
        _require_report_input(source_report, key="compile_report", path=compile_path)
        compile_report = read_checked_json(compile_path)
        reconstructed = build_data_invalid_source_report(
            manifest=manifest,
            reason=str(source_report.get("reason", "")),
            compile_report=compile_report,
            prefit_evidence=_mapping(source_report.get("prefit_evidence")),
            inputs=_mapping(source_report.get("inputs")),
        )
        _compare_reconstructed_fields(
            source_report,
            reconstructed,
            fields=(
                "format_version",
                "phase",
                "status",
                "verdict",
                "manifest_checksum",
                "precollection_aborted_actions",
                "maximum_new_actions",
                "checks",
                "passed",
                "firewall",
                "compile_report_checksum",
            ),
            label="source report",
        )
        return dict(source_report)

    metrics = source_report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ManifestDriftError("source report lacks reconstructable metrics")
    fresh_path = destination / "source_events.jsonl"
    replay_path = destination / "replay_events.jsonl"
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    _require_report_input(source_report, key="fresh_events", path=fresh_path)
    _require_report_input(source_report, key="replay_events", path=replay_path)
    _require_report_input(source_report, key="cross_fit_audit", path=cross_fit_path)
    _derive_source_control_views(metrics)
    _verify_source_evidence_binding(
        metrics,
        manifest=manifest,
        fresh_path=fresh_path,
        replay_path=replay_path,
        cross_fit_path=cross_fit_path,
    )
    verified_metrics = dict(metrics)
    verified_metrics["challenger_recipe"] = _verified_challenger_recipe_binding(
        metrics,
        output_dir=destination,
        manifest=manifest,
        limits=limits,
    )
    reconstructed = build_source_gate_report(
        manifest=manifest,
        metrics=verified_metrics,
    )
    _compare_reconstructed_fields(
        source_report,
        reconstructed,
        fields=(
            "format_version",
            "phase",
            "status",
            "verdict",
            "manifest_checksum",
            "precollection_aborted_actions",
            "maximum_new_actions",
            "metrics",
            "registered_controls",
            "checks",
            "passed",
            "firewall",
        ),
        label="source report",
    )
    return dict(source_report)


def _read_canonical_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestDriftError(
                f"invalid validation JSONL row {line_number}"
            ) from exc
        if not isinstance(row, dict) or raw != canonical_json(row):
            raise ManifestDriftError(f"noncanonical validation JSONL row {line_number}")
        rows.append(row)
    return rows


def _reconstruct_validation_report(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_path: Path,
    validation_report: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    _require_report_input(validation_report, key="source_report", path=source_path)
    runs_path = destination / "validation_runs.jsonl"
    _require_report_input(validation_report, key="validation_runs", path=runs_path)
    timing_path = destination / VALIDATION_TIMING_PROOF_FILENAME
    _require_report_input(
        validation_report,
        key="validation_timing_proof",
        path=timing_path,
    )
    runs = _read_canonical_jsonl_objects(runs_path)
    reconstructed_metrics = aggregate_validation_runs(runs)
    stored_metrics = validation_report.get("metrics")
    if not isinstance(stored_metrics, Mapping):
        raise ManifestDriftError("validation report lacks reconstructable metrics")
    reported_wall = reconstructed_metrics["wall_seconds"]
    if stored_metrics.get("reported_pair_wall_seconds") != reported_wall:
        raise ManifestDriftError("validation reported pair wall time drifted")
    timing_proof = _read_validation_timing_proof(
        path=timing_path,
        manifest=manifest,
        source_report=source_report,
        source_path=source_path,
        runs_path=runs_path,
        runs=runs,
        reported_pair_wall_seconds=float(reported_wall),
    )
    reconstructed_metrics["reported_pair_wall_seconds"] = reported_wall
    reconstructed_metrics["wall_seconds"] = timing_proof["monotonic_elapsed_seconds"]
    if canonical_json(dict(stored_metrics)) != canonical_json(reconstructed_metrics):
        raise ManifestDriftError(
            "validation metrics drifted from validation_runs.jsonl"
        )
    reconstructed = build_validation_report(
        manifest=manifest,
        source_report=source_report,
        metrics=reconstructed_metrics,
    )
    _compare_reconstructed_fields(
        validation_report,
        reconstructed,
        fields=(
            "format_version",
            "phase",
            "status",
            "verdict",
            "manifest_checksum",
            "source_report_checksum",
            "metrics",
            "checks",
            "passed",
            "firewall",
        ),
        label="validation report",
    )
    return dict(validation_report)


def build_final_report(
    *,
    manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    validation_report: Mapping[str, Any] | None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_passed = source_report.get("status") == "PASS_T10_2_SOURCE_GATE"
    if source_passed:
        if validation_report is None:
            raise GateRefusalError("a passing source gate requires validation report")
        verdict = str(validation_report.get("verdict", ""))
        supported = validation_report.get("status") == "PASS_T10_2_VALIDATION"
    else:
        verdict = str(source_report.get("verdict", "DATA_OR_PROVENANCE_INVALID"))
        supported = False
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "report",
        "status": "T10_2_COMPLETE",
        "verdict": verdict,
        "supported": supported,
        "manifest_checksum": manifest["manifest_checksum"],
        "source_report_checksum": source_report["report_checksum"],
        "validation_report_checksum": (
            None if validation_report is None else validation_report["report_checksum"]
        ),
        "inputs": dict(inputs or {}),
        "artifact_inventory": {
            "inventory_sidecar": f"../{ARTIFACT_INVENTORY_FILENAME}",
            "binding_sidecar": f"../{REPORT_INVENTORY_BINDING_FILENAME}",
            "cyclic_hash_dependency": False,
        },
        "firewall": {
            "source_validation_opened": source_passed,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
        "evidence_boundary": (
            "Factor scores and correspondence are diagnostic only; the verdict "
            "is determined by registered causal controls and active utility."
        ),
    }
    return signed_payload(payload, checksum_key="report_checksum")


def _write_report_inventory_sidecars(
    *,
    manifest: Mapping[str, Any],
    destination: Path,
    report_path: Path,
    limits: ResourceLimits,
) -> None:
    from .t10_2_artifact_inventory import build_inventory, write_inventory

    inventory_path = destination.parent / ARTIFACT_INVENTORY_FILENAME
    binding_path = destination.parent / REPORT_INVENTORY_BINDING_FILENAME
    inventory = build_inventory(
        repository_root=destination.parent,
        artifact_root=destination.name,
        output_path=ARTIFACT_INVENTORY_FILENAME,
    )
    write_inventory(
        inventory,
        repository_root=destination.parent,
        output_path=ARTIFACT_INVENTORY_FILENAME,
    )
    binding = signed_payload(
        {
            "format_version": "sage-t10.2-report-inventory-binding-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "report": artifact_descriptor(report_path),
            "inventory": artifact_descriptor(inventory_path),
            "inventory_checksum": inventory["inventory_checksum"],
            "cycle_free": True,
        },
        checksum_key="binding_checksum",
    )
    write_compact_json(binding_path, binding)
    enforce_artifact_limit(inventory_path, kind="derived", limits=limits)
    enforce_artifact_limit(binding_path, kind="derived", limits=limits)


def report_phase(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_report_path: str | Path | None = None,
    validation_report_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    """Rebuild the final report deterministically and byte-identically."""

    root = Path(repo_root or _repo_root()).resolve()
    manifest = load_manifest(manifest_path, repo_root=root)
    destination = Path(output_dir)
    source_path = Path(source_report_path or destination / "source_report.json")
    source_report = read_checked_json(source_path)
    if source_report.get("manifest_checksum") != manifest["manifest_checksum"]:
        raise ManifestDriftError("source report/manifest binding drifted")
    source_report = _reconstruct_source_report(
        manifest=manifest,
        source_report=source_report,
        destination=destination,
        limits=limits,
    )
    source_passed = source_report.get("status") == "PASS_T10_2_SOURCE_GATE"
    if source_passed:
        source_report = require_source_gate(
            source_report_path=source_path,
            manifest=manifest,
            output_dir=destination,
            limits=limits,
        )
    validation_report: dict[str, Any] | None = None
    validation_path = Path(
        validation_report_path or destination / "validation_report.json"
    )
    if source_passed:
        validation_report = read_checked_json(validation_path)
        if validation_report.get("manifest_checksum") != manifest["manifest_checksum"]:
            raise ManifestDriftError("validation report/manifest binding drifted")
        if validation_report.get("source_report_checksum") != source_report.get(
            "report_checksum"
        ):
            raise ManifestDriftError("validation/source report binding drifted")
        validation_report = _reconstruct_validation_report(
            manifest=manifest,
            source_report=source_report,
            source_path=source_path,
            validation_report=validation_report,
            destination=destination,
        )
    inputs: dict[str, Any] = {
        "source_report": artifact_descriptor(source_path),
    }
    if validation_report is not None:
        inputs["validation_report"] = artifact_descriptor(validation_path)
    report = build_final_report(
        manifest=manifest,
        source_report=source_report,
        validation_report=validation_report,
        inputs=inputs,
    )
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    write_compact_json(report_path, report)
    _write_report_inventory_sidecars(
        manifest=manifest,
        destination=destination,
        report_path=report_path,
        limits=limits,
    )
    enforce_output_artifacts(destination, limits=limits)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-report")
    parser.add_argument("--validation-report")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--skip-repository-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    try:
        if args.phase == "freeze":
            payload = freeze_manifest(
                output_path=args.manifest,
                repo_root=args.repo_root,
                verify_repository=not args.skip_repository_check,
            )
        elif args.phase == "collect":
            # Import only after the frozen manifest and source firewall are in
            # place.  The adapter itself remains lazy and cannot open an ARC
            # environment before ``collect_phase`` authorizes a source lane.
            from .t10_2_runtime import T10_2SourceFactory

            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = collect_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
                env_factory=T10_2SourceFactory(manifest=manifest),
            )
        elif args.phase == "compile":
            payload = compile_phase(
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
            )
        elif args.phase == "replay":
            # The CLI admits no arbitrary replay path.  It reconstructs the
            # only registered ledger from the three manifest-bound V4.3
            # source shards, then the normal replay phase verifies it again.
            from .t10_2_runtime import build_v4_3_replay_ledger

            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            with tempfile.TemporaryDirectory(prefix="sage-t10-2-replay-") as temporary:
                replay_input = Path(temporary) / "manifest_bound_replay.jsonl"
                build_v4_3_replay_ledger(
                    manifest=manifest,
                    repo_root=args.repo_root,
                    output_path=replay_input,
                )
                payload = replay_phase(
                    replay_input_path=replay_input,
                    manifest_path=args.manifest,
                    output_dir=output,
                    repo_root=args.repo_root,
                )
        elif args.phase == "source-train":
            # Metrics are produced only by the code-bound trainer over the
            # checksummed fresh and replay ledgers.  There is deliberately no
            # CLI surface for injecting a JSON score report.
            from .t10_2_runtime import run_source_trainer

            payload = source_train_phase(
                trainer=run_source_trainer,
                manifest_path=args.manifest,
                output_dir=output,
                repo_root=args.repo_root,
            )
        elif args.phase == "validate":
            from .t10_2_runtime import (
                T10_1BehaviorFrozenPolicyFactory,
                T10_2GaugePolicyFactory,
                T10_2ValidationFactory,
            )

            source_path = Path(args.source_report or output / "source_report.json")
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            authorized_source = require_source_gate(
                source_report_path=source_path,
                manifest=manifest,
                output_dir=output,
            )
            baseline_policy_factory = T10_1BehaviorFrozenPolicyFactory(
                repo_root=args.repo_root,
            )
            challenger_policy_factory = T10_2GaugePolicyFactory(
                source_report=authorized_source,
                manifest=manifest,
                output_dir=output,
            )
            payload = validate_phase(
                manifest_path=args.manifest,
                output_dir=output,
                source_report_path=source_path,
                repo_root=args.repo_root,
                env_factory=T10_2ValidationFactory(
                    source_report=authorized_source,
                    manifest=manifest,
                    t10_1_policy_factory=baseline_policy_factory,
                    t10_2_policy_factory=challenger_policy_factory,
                ),
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
                    "status": "FAIL_CLOSED",
                }
            )
        )
        return 2
    print(canonical_json(payload))
    if payload.get("passed") is False:
        return 2
    return 0


__all__ = [
    "AR25_GAME",
    "BASELINE_COMMIT",
    "BASELINE_FROZEN_SHA256",
    "CHALLENGER_RECIPE_FILENAME",
    "COMPACT_PROJECTION_FORMAT_VERSION",
    "COMPACT_QUOTIENT_FORMAT_VERSION",
    "CONFIRMATION_SEEDS",
    "DEFAULT_CODE_FILES",
    "DEFAULT_INPUT_FILES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RESOURCE_LIMITS",
    "DISCOVERY_SEEDS",
    "EVENT_FORMAT_VERSION",
    "FORMAT_VERSION",
    "MAXIMUM_COMPACT_EVENT_BYTES",
    "MAXIMUM_MODEL_VIEW_BYTES",
    "PHASES",
    "REGISTERED_FRAME_ORDER",
    "REGISTERED_SOURCE_CONTROLS",
    "SOURCE_GAMES",
    "SOURCE_MAXIMUM_ACTIONS",
    "SOURCE_MAXIMUM_NEW_ACTIONS",
    "SOURCE_PRECOLLECTION_ABORTED_ACTIONS",
    "VALIDATION_GAMES",
    "VALIDATION_SEEDS",
    "DataGateError",
    "FirewallError",
    "GateRefusalError",
    "ManifestDriftError",
    "ResourceGateError",
    "ResourceLimits",
    "ResourceSnapshot",
    "RuntimeUnavailableError",
    "aggregate_validation_runs",
    "artifact_descriptor",
    "build_data_invalid_source_report",
    "build_final_report",
    "build_manifest",
    "build_parser",
    "build_qa_report",
    "build_source_gate_report",
    "build_validation_report",
    "canonical_json",
    "canonical_json_file_sha256",
    "canonical_sha256",
    "collect_phase",
    "compile_phase",
    "enforce_environment_firewall",
    "enforce_resource_limits",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "main",
    "paired_bootstrap_lower",
    "read_checked_json",
    "read_event_ledger",
    "replay_phase",
    "report_phase",
    "require_source_gate",
    "seal_event",
    "signed_payload",
    "source_train_phase",
    "validate_phase",
    "validate_prefit_evidence",
    "validate_source_events",
    "write_compact_json",
    "write_event_ledger",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
