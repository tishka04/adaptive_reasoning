"""Durable, resumable active-source runtime for SAGE.T10.2.1.

T10.2.1 is an execution-only amendment.  The scientific representation,
posterior, grammar, observer frames, transports, compact physical-event schema,
and decision engine remain the frozen T10.2 implementations imported below.
This module adds a process boundary around each reset and a write-ahead journal
around every physical action.

The central accounting invariant is intentionally conservative::

    authorized intents == sealed physical events + explicitly unresolved intents

An intent is made durable by the parent process before the child is allowed to
call the environment's ``step`` method.  A child death after that acknowledgement
therefore leaves an explicit unresolved intent instead of silently losing an
action.  Such a collection is never admitted to fitting.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from . import t10_2_protocol as _t10_2_protocol
from . import t10_2_runtime as _science
from .factorized_posterior_v10_2 import (
    FactorizedCandidateBank,
    FactorizedGaugeProgramPosterior,
)
from .gauge_inference_v10_2 import (
    GaugeDecisionEngine,
    GaugeProgramPosterior,
    JointGaugeHypothesis,
    rank_option_sequence_signatures,
)
from .mixed_automata_v10_2 import generate_mixed_grammar, repeat
from .observer_frames_v10_2 import OBSERVER_FRAME_SPECS, PhysicalEventBundle


FORMAT_VERSION = "sage-t10.2.1-runtime-v1"
COLLECTION_FORMAT_VERSION = "sage-t10.2.1-protocol-v1"
JOURNAL_FORMAT_VERSION = "sage-t10.2.1-durable-journal-v1"
LANE_REPORT_FORMAT_VERSION = "sage-t10.2.1-lane-report-v1"
RESET_REPORT_FORMAT_VERSION = "sage-t10.2.1-reset-report-v1"
CHECKPOINT_FORMAT_VERSION = "sage-t10.2.1-collection-checkpoint-v1"
CROSS_FIT_AUDIT_FORMAT_VERSION = "sage-t10.2.1-cross-fit-audit-v1"

SOURCE_GAMES = tuple(_t10_2_protocol.SOURCE_GAMES)
DISCOVERY_SEEDS = (101, 102, 103)
CONFIRMATION_SEEDS = (111, 112, 113)
SOURCE_RESETS_PER_LANE = 4
SOURCE_ACTIONS_PER_RESET = 64
SOURCE_MAXIMUM_AUTHORIZED_INTENTS = (
    len(SOURCE_GAMES)
    * (len(DISCOVERY_SEEDS) + len(CONFIRMATION_SEEDS))
    * SOURCE_RESETS_PER_LANE
    * SOURCE_ACTIONS_PER_RESET
)
RESET_COOPERATIVE_SECONDS = 55.0
RESET_HARD_SECONDS = 60.0
LANE_FINALIZATION_SECONDS = 10.0
LANE_HARD_SECONDS = 250.0
COLLECTION_COOPERATIVE_SECONDS = 5_100.0
COLLECTION_ABSOLUTE_SECONDS = 5_400.0
FIT_SEED = 10_201
BOOTSTRAP_SEED = 10_202
PERMUTATION_SEED = 10_203

DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "t10_2_1_durable_gauge_posterior"
)
SOURCE_EVENTS_FILENAME = "source_events.jsonl"
CROSS_FIT_AUDIT_FILENAME = "cross_fit_audit.json"
COLLECTION_REPORT_FILENAME = "collection_report.json"
CHECKPOINT_FILENAME = "source_collection_checkpoint.json"
JOURNAL_DIRECTORY_NAME = "source_collection_journal"
INVOCATION_STATE_FILENAME = "collection_invocation_state.json"
INVOCATION_STATE_FORMAT_VERSION = "sage-t10.2.1-collection-invocation-v1"
INVOCATION_TERMINAL_FILENAME = "collection_invocation_terminal.json"
INVOCATION_TERMINAL_FORMAT_VERSION = (
    "sage-t10.2.1-collection-invocation-terminal-v1"
)

LaneSplit = Literal["discovery", "leave_one_game_out_confirmation"]
ReportStatus = Literal[
    "PENDING", "RUNNING", "COMPLETE", "ABORTED", "UNATTESTABLE"
]


class T10_2_1RuntimeError(RuntimeError):
    """Base class for fail-closed T10.2.1 runtime errors."""


class JournalIntegrityError(T10_2_1RuntimeError):
    """A durable record is malformed, unsigned, truncated, or inconsistent."""


class JournalConflictError(JournalIntegrityError):
    """An immutable journal key was reused with different canonical content."""


class ResumeRefusalError(T10_2_1RuntimeError):
    """A partial physical reset cannot be replayed safely."""


class WorkerProtocolError(T10_2_1RuntimeError):
    """The reset worker violated the intent/event/update handshake."""


class WorkerTimeoutError(T10_2_1RuntimeError):
    """A reset worker exceeded a registered cooperative or hard deadline."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON refuses non-finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _signed(payload: Mapping[str, Any], *, checksum_key: str) -> dict[str, Any]:
    result = dict(_jsonable(payload))
    result.pop(checksum_key, None)
    result[checksum_key] = canonical_sha256(result)
    return result


def _verify_signed(payload: Mapping[str, Any], *, checksum_key: str) -> None:
    unsigned = dict(payload)
    observed = str(unsigned.pop(checksum_key, ""))
    if not observed or observed != canonical_sha256(unsigned):
        raise JournalIntegrityError(f"invalid {checksum_key}")


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _require_digest(value: Any, *, label: str, allow_empty: bool = False) -> str:
    digest = str(value)
    if allow_empty and not digest:
        return digest
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _status(value: Any) -> ReportStatus:
    candidate = str(value)
    if candidate not in {"PENDING", "RUNNING", "COMPLETE", "ABORTED", "UNATTESTABLE"}:
        raise ValueError(f"invalid durable status: {candidate}")
    return candidate  # type: ignore[return-value]


@dataclass(frozen=True)
class SourceLaneKey:
    split: LaneSplit
    game_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.split not in {"discovery", "leave_one_game_out_confirmation"}:
            raise ValueError(f"invalid source split: {self.split}")
        if self.game_id not in SOURCE_GAMES:
            raise ValueError(f"non-source game: {self.game_id}")
        seeds = DISCOVERY_SEEDS if self.split == "discovery" else CONFIRMATION_SEEDS
        if isinstance(self.seed, bool) or int(self.seed) not in seeds:
            raise ValueError(f"seed {self.seed} is not registered for {self.split}")

    @property
    def lane_id(self) -> str:
        return canonical_sha256(self.identity_dict())

    def identity_dict(self) -> dict[str, Any]:
        return {"split": self.split, "game_id": self.game_id, "seed": int(self.seed)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "lane_id": self.lane_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceLaneKey":
        lane = cls(
            split=str(payload.get("split", "")),  # type: ignore[arg-type]
            game_id=str(payload.get("game_id", "")),
            seed=int(payload.get("seed", -1)),
        )
        if payload.get("lane_id") not in (None, lane.lane_id):
            raise JournalIntegrityError("source lane id drifted")
        return lane


def confirmation_controller_order(seed: int) -> tuple[str, ...]:
    if int(seed) not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation controller order requires a confirmation seed")
    forward = (
        "learned",
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
    )
    return forward if int(seed) % 2 == 0 else tuple(reversed(forward))


@dataclass(frozen=True)
class ResetWorkSpec:
    lane: SourceLaneKey
    reset_index: int
    controller: str
    maximum_actions: int = SOURCE_ACTIONS_PER_RESET
    cooperative_seconds: float = RESET_COOPERATIVE_SECONDS
    hard_seconds: float = RESET_HARD_SECONDS
    spec_checksum: str = ""

    def __post_init__(self) -> None:
        reset_index = _require_nonnegative_int(self.reset_index, label="reset index")
        if reset_index >= SOURCE_RESETS_PER_LANE:
            raise ValueError("reset index exceeds the registered four-reset lane")
        expected = (
            "balanced_discovery"
            if self.lane.split == "discovery"
            else confirmation_controller_order(self.lane.seed)[reset_index]
        )
        if self.controller != expected:
            raise ValueError(
                f"controller {self.controller!r} does not match registered {expected!r}"
            )
        if self.maximum_actions != SOURCE_ACTIONS_PER_RESET:
            raise ValueError("T10.2.1 requires exactly 64 authorized actions per reset")
        if float(self.cooperative_seconds) != RESET_COOPERATIVE_SECONDS:
            raise ValueError("T10.2.1 cooperative reset deadline drifted")
        if float(self.hard_seconds) != RESET_HARD_SECONDS:
            raise ValueError("T10.2.1 hard reset deadline drifted")
        expected_checksum = canonical_sha256(self.unsigned_dict())
        if self.spec_checksum and self.spec_checksum != expected_checksum:
            raise JournalIntegrityError("reset work checksum drifted")
        object.__setattr__(self, "spec_checksum", expected_checksum)

    @property
    def work_id(self) -> str:
        return canonical_sha256(
            {"lane_id": self.lane.lane_id, "reset_index": self.reset_index}
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.to_dict(),
            "reset_index": self.reset_index,
            "controller": self.controller,
            "maximum_actions": self.maximum_actions,
            "cooperative_seconds": float(self.cooperative_seconds),
            "hard_seconds": float(self.hard_seconds),
            "work_id": self.work_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "spec_checksum": self.spec_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResetWorkSpec":
        work = cls(
            lane=SourceLaneKey.from_dict(_mapping(payload.get("lane"))),
            reset_index=int(payload.get("reset_index", -1)),
            controller=str(payload.get("controller", "")),
            maximum_actions=int(payload.get("maximum_actions", -1)),
            cooperative_seconds=float(payload.get("cooperative_seconds", -1.0)),
            hard_seconds=float(payload.get("hard_seconds", -1.0)),
            spec_checksum=str(payload.get("spec_checksum", "")),
        )
        if payload.get("work_id") not in (None, work.work_id):
            raise JournalIntegrityError("reset work id drifted")
        return work


def reset_work_specs(lane: SourceLaneKey) -> tuple[ResetWorkSpec, ...]:
    controllers = (
        ("balanced_discovery",) * SOURCE_RESETS_PER_LANE
        if lane.split == "discovery"
        else confirmation_controller_order(lane.seed)
    )
    return tuple(
        ResetWorkSpec(lane=lane, reset_index=index, controller=controller)
        for index, controller in enumerate(controllers)
    )


def source_lane_registry() -> tuple[SourceLaneKey, ...]:
    lanes = tuple(
        SourceLaneKey("discovery", game_id, seed)
        for game_id in SOURCE_GAMES
        for seed in DISCOVERY_SEEDS
    )
    return lanes + tuple(
        SourceLaneKey("leave_one_game_out_confirmation", game_id, seed)
        for game_id in SOURCE_GAMES
        for seed in CONFIRMATION_SEEDS
    )


def _action_commitment(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping) and set(action) == {
        "name",
        "parameter_arity",
        "grounding_sha256",
    }:
        name = str(action["name"])
        arity = _require_nonnegative_int(
            action["parameter_arity"], label="action parameter arity"
        )
        grounding = _require_digest(
            action["grounding_sha256"], label="action grounding commitment"
        )
    else:
        name = _science._action_name(action)
        data = _science._action_data(action)
        arity = len(data)
        grounding = canonical_sha256({"name": name, "data": data})
    if not name:
        raise ValueError("action commitment requires a non-empty action name")
    return {
        "name": name,
        "parameter_arity": arity,
        "grounding_sha256": grounding,
    }


@dataclass(frozen=True)
class ActionIntent:
    lane: SourceLaneKey
    reset_index: int
    step_index: int
    action: Mapping[str, Any]
    manifest_checksum: str
    intent_id: str = ""
    intent_checksum: str = ""

    def __post_init__(self) -> None:
        if not 0 <= int(self.reset_index) < SOURCE_RESETS_PER_LANE:
            raise ValueError("action intent reset index is invalid")
        if not 0 <= int(self.step_index) < SOURCE_ACTIONS_PER_RESET:
            raise ValueError("action intent step index is invalid")
        _require_digest(self.manifest_checksum, label="manifest checksum")
        object.__setattr__(self, "action", _action_commitment(self.action))
        expected_id = canonical_sha256(
            {
                "manifest_checksum": self.manifest_checksum,
                "lane_id": self.lane.lane_id,
                "reset_index": int(self.reset_index),
                "step_index": int(self.step_index),
            }
        )
        if self.intent_id and self.intent_id != expected_id:
            raise JournalIntegrityError("action intent id drifted")
        object.__setattr__(self, "intent_id", expected_id)
        expected_checksum = canonical_sha256(self.unsigned_dict())
        if self.intent_checksum and self.intent_checksum != expected_checksum:
            raise JournalIntegrityError("action intent checksum drifted")
        object.__setattr__(self, "intent_checksum", expected_checksum)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": JOURNAL_FORMAT_VERSION,
            "kind": "action_intent",
            "intent_id": self.intent_id,
            "lane": self.lane.to_dict(),
            "reset_index": int(self.reset_index),
            "step_index": int(self.step_index),
            "action": dict(self.action),
            "manifest_checksum": self.manifest_checksum,
            "charged_action": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "intent_checksum": self.intent_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionIntent":
        if payload.get("kind") != "action_intent":
            raise JournalIntegrityError("journal record is not an action intent")
        return cls(
            lane=SourceLaneKey.from_dict(_mapping(payload.get("lane"))),
            reset_index=int(payload.get("reset_index", -1)),
            step_index=int(payload.get("step_index", -1)),
            action=_mapping(payload.get("action")),
            manifest_checksum=str(payload.get("manifest_checksum", "")),
            intent_id=str(payload.get("intent_id", "")),
            intent_checksum=str(payload.get("intent_checksum", "")),
        )


@dataclass(frozen=True)
class PhysicalEventReceipt:
    intent_id: str
    intent_checksum: str
    event_id: str
    event_checksum: str
    event_payload_sha256: str
    receipt_checksum: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.intent_id, "intent id"),
            (self.intent_checksum, "intent checksum"),
            (self.event_checksum, "event checksum"),
            (self.event_payload_sha256, "event payload checksum"),
        ):
            _require_digest(value, label=label)
        if not str(self.event_id):
            raise ValueError("physical event receipt requires an event id")
        expected = canonical_sha256(self.unsigned_dict())
        if self.receipt_checksum and self.receipt_checksum != expected:
            raise JournalIntegrityError("physical event receipt checksum drifted")
        object.__setattr__(self, "receipt_checksum", expected)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": JOURNAL_FORMAT_VERSION,
            "kind": "physical_event_receipt",
            "intent_id": self.intent_id,
            "intent_checksum": self.intent_checksum,
            "event_id": self.event_id,
            "event_checksum": self.event_checksum,
            "event_payload_sha256": self.event_payload_sha256,
            "charged_action": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_checksum": self.receipt_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalEventReceipt":
        return cls(
            intent_id=str(payload.get("intent_id", "")),
            intent_checksum=str(payload.get("intent_checksum", "")),
            event_id=str(payload.get("event_id", "")),
            event_checksum=str(payload.get("event_checksum", "")),
            event_payload_sha256=str(payload.get("event_payload_sha256", "")),
            receipt_checksum=str(payload.get("receipt_checksum", "")),
        )


@dataclass(frozen=True)
class PosteriorUpdateReceipt:
    intent_id: str
    event_checksum: str
    status: Literal["APPLIED", "SKIPPED", "REFUSED"]
    posterior_state_sha256: str | None = None
    error_kind: str | None = None
    receipt_checksum: str = ""

    def __post_init__(self) -> None:
        _require_digest(self.intent_id, label="intent id")
        _require_digest(self.event_checksum, label="event checksum")
        if self.status not in {"APPLIED", "SKIPPED", "REFUSED"}:
            raise ValueError(f"invalid posterior update status: {self.status}")
        if self.posterior_state_sha256 is not None:
            _require_digest(
                self.posterior_state_sha256, label="posterior state checksum"
            )
        if self.status == "REFUSED" and not self.error_kind:
            raise ValueError("refused posterior update requires an error kind")
        expected = canonical_sha256(self.unsigned_dict())
        if self.receipt_checksum and self.receipt_checksum != expected:
            raise JournalIntegrityError("posterior update receipt checksum drifted")
        object.__setattr__(self, "receipt_checksum", expected)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": JOURNAL_FORMAT_VERSION,
            "kind": "posterior_update_receipt",
            "intent_id": self.intent_id,
            "event_checksum": self.event_checksum,
            "status": self.status,
            "posterior_state_sha256": self.posterior_state_sha256,
            "error_kind": self.error_kind,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_checksum": self.receipt_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PosteriorUpdateReceipt":
        return cls(
            intent_id=str(payload.get("intent_id", "")),
            event_checksum=str(payload.get("event_checksum", "")),
            status=str(payload.get("status", "")),  # type: ignore[arg-type]
            posterior_state_sha256=(
                None
                if payload.get("posterior_state_sha256") is None
                else str(payload.get("posterior_state_sha256"))
            ),
            error_kind=(
                None if payload.get("error_kind") is None else str(payload["error_kind"])
            ),
            receipt_checksum=str(payload.get("receipt_checksum", "")),
        )


@dataclass(frozen=True)
class UnresolvedIntentReceipt:
    intent_id: str
    intent_checksum: str
    reason: Literal[
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "worker_failed",
        "parent_interrupted",
        "environment_call_unattestable",
    ]
    receipt_checksum: str = ""

    def __post_init__(self) -> None:
        _require_digest(self.intent_id, label="intent id")
        _require_digest(self.intent_checksum, label="intent checksum")
        if self.reason not in {
            "hard_reset_timeout",
            "cooperative_reset_deadline",
            "worker_failed",
            "parent_interrupted",
            "environment_call_unattestable",
        }:
            raise ValueError(f"invalid unresolved-intent reason: {self.reason}")
        expected = canonical_sha256(self.unsigned_dict())
        if self.receipt_checksum and self.receipt_checksum != expected:
            raise JournalIntegrityError("unresolved-intent receipt checksum drifted")
        object.__setattr__(self, "receipt_checksum", expected)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": JOURNAL_FORMAT_VERSION,
            "kind": "unresolved_intent_receipt",
            "intent_id": self.intent_id,
            "intent_checksum": self.intent_checksum,
            "reason": self.reason,
            "charged_action": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "receipt_checksum": self.receipt_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnresolvedIntentReceipt":
        if payload.get("kind") != "unresolved_intent_receipt":
            raise JournalIntegrityError("record is not an unresolved-intent receipt")
        return cls(
            intent_id=str(payload.get("intent_id", "")),
            intent_checksum=str(payload.get("intent_checksum", "")),
            reason=str(payload.get("reason", "")),  # type: ignore[arg-type]
            receipt_checksum=str(payload.get("receipt_checksum", "")),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JournalIntegrityError("expected a JSON object")
    return value


def _schema_count_key(name: str, arity: int) -> str:
    return canonical_json([str(name), int(arity)])


def _schema_count_from_key(value: str) -> tuple[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise JournalIntegrityError("invalid schema-count key") from exc
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise JournalIntegrityError("invalid schema-count key")
    return str(parsed[0]), int(parsed[1])


@dataclass(frozen=True)
class ResetReport:
    work: ResetWorkSpec
    status: ReportStatus
    issued_intents: int
    sealed_events: int
    unresolved_intents: int
    posterior_updates: int
    elapsed_seconds: float
    stop_reason: str
    event_ids_sha256: str
    continuation: Mapping[str, Any]
    report_checksum: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        issued = _require_nonnegative_int(self.issued_intents, label="issued intents")
        sealed = _require_nonnegative_int(self.sealed_events, label="sealed events")
        unresolved = _require_nonnegative_int(
            self.unresolved_intents, label="unresolved intents"
        )
        updates = _require_nonnegative_int(
            self.posterior_updates, label="posterior updates"
        )
        if issued != sealed + unresolved:
            raise JournalIntegrityError(
                "reset accounting violates intents = sealed + unresolved"
            )
        if issued > self.work.maximum_actions or updates > sealed:
            raise JournalIntegrityError("reset accounting exceeds its registered bounds")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("reset elapsed time must be finite and non-negative")
        if self.status == "COMPLETE" and unresolved:
            raise JournalIntegrityError("a complete reset cannot contain unresolved intents")
        _require_digest(self.event_ids_sha256, label="reset event-id root")
        expected = canonical_sha256(self.unsigned_dict())
        if self.report_checksum and self.report_checksum != expected:
            raise JournalIntegrityError("reset report checksum drifted")
        object.__setattr__(self, "report_checksum", expected)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": RESET_REPORT_FORMAT_VERSION,
            "work": self.work.to_dict(),
            "status": self.status,
            "issued_intents": self.issued_intents,
            "sealed_events": self.sealed_events,
            "unresolved_intents": self.unresolved_intents,
            "posterior_updates": self.posterior_updates,
            "elapsed_seconds": float(self.elapsed_seconds),
            "stop_reason": self.stop_reason,
            "event_ids_sha256": self.event_ids_sha256,
            "continuation": _jsonable(self.continuation),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_checksum": self.report_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResetReport":
        return cls(
            work=ResetWorkSpec.from_dict(_mapping(payload.get("work"))),
            status=str(payload.get("status", "")),  # type: ignore[arg-type]
            issued_intents=int(payload.get("issued_intents", -1)),
            sealed_events=int(payload.get("sealed_events", -1)),
            unresolved_intents=int(payload.get("unresolved_intents", -1)),
            posterior_updates=int(payload.get("posterior_updates", -1)),
            elapsed_seconds=float(payload.get("elapsed_seconds", -1.0)),
            stop_reason=str(payload.get("stop_reason", "")),
            event_ids_sha256=str(payload.get("event_ids_sha256", "")),
            continuation=_mapping(payload.get("continuation", {})),
            report_checksum=str(payload.get("report_checksum", "")),
        )


@dataclass(frozen=True)
class LaneReport:
    lane: SourceLaneKey
    status: ReportStatus
    resets: tuple[ResetReport, ...]
    issued_intents: int
    sealed_events: int
    unresolved_intents: int
    elapsed_seconds: float
    cross_fit_unit: Mapping[str, Any] | None = None
    report_checksum: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        ordered = tuple(sorted(self.resets, key=lambda item: item.work.reset_index))
        if ordered != self.resets or any(item.work.lane != self.lane for item in ordered):
            raise JournalIntegrityError("lane report reset registry drifted")
        if len({item.work.reset_index for item in ordered}) != len(ordered):
            raise JournalIntegrityError("lane report contains duplicate resets")
        issued = sum(item.issued_intents for item in ordered)
        sealed = sum(item.sealed_events for item in ordered)
        unresolved = sum(item.unresolved_intents for item in ordered)
        if (self.issued_intents, self.sealed_events, self.unresolved_intents) != (
            issued,
            sealed,
            unresolved,
        ):
            raise JournalIntegrityError("lane report action accounting drifted")
        if issued != sealed + unresolved:
            raise JournalIntegrityError(
                "lane accounting violates intents = sealed + unresolved"
            )
        if self.status == "COMPLETE" and (
            len(ordered) != SOURCE_RESETS_PER_LANE
            or unresolved
            or any(item.status != "COMPLETE" for item in ordered)
        ):
            raise JournalIntegrityError("complete lane lacks four complete safe resets")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("lane elapsed time must be finite and non-negative")
        expected = canonical_sha256(self.unsigned_dict())
        if self.report_checksum and self.report_checksum != expected:
            raise JournalIntegrityError("lane report checksum drifted")
        object.__setattr__(self, "report_checksum", expected)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": LANE_REPORT_FORMAT_VERSION,
            "lane": self.lane.to_dict(),
            "status": self.status,
            "resets": [item.to_dict() for item in self.resets],
            "issued_intents": self.issued_intents,
            "sealed_events": self.sealed_events,
            "unresolved_intents": self.unresolved_intents,
            "elapsed_seconds": float(self.elapsed_seconds),
            "cross_fit_unit": (
                None if self.cross_fit_unit is None else _jsonable(self.cross_fit_unit)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_checksum": self.report_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneReport":
        raw_resets = payload.get("resets")
        if not isinstance(raw_resets, list):
            raise JournalIntegrityError("lane report resets must be a list")
        raw_cross_fit = payload.get("cross_fit_unit")
        return cls(
            lane=SourceLaneKey.from_dict(_mapping(payload.get("lane"))),
            status=str(payload.get("status", "")),  # type: ignore[arg-type]
            resets=tuple(ResetReport.from_dict(_mapping(item)) for item in raw_resets),
            issued_intents=int(payload.get("issued_intents", -1)),
            sealed_events=int(payload.get("sealed_events", -1)),
            unresolved_intents=int(payload.get("unresolved_intents", -1)),
            elapsed_seconds=float(payload.get("elapsed_seconds", -1.0)),
            cross_fit_unit=(
                None if raw_cross_fit is None else _mapping(raw_cross_fit)
            ),
            report_checksum=str(payload.get("report_checksum", "")),
        )


@dataclass(frozen=True)
class CollectionCheckpoint:
    manifest_checksum: str
    lane_registry_sha256: str
    lane_reports: tuple[LaneReport, ...]
    cumulative_active_seconds: float
    journal_reconstructed: bool
    checkpoint_reconstructed: bool
    physical_steps_replayed_on_resume: int
    revision: int
    open_lane_id: str | None = None
    open_lane_elapsed_seconds: float = 0.0
    checkpoint_checksum: str = ""

    def __post_init__(self) -> None:
        _require_digest(self.manifest_checksum, label="manifest checksum")
        _require_digest(self.lane_registry_sha256, label="lane registry checksum")
        _require_nonnegative_int(self.revision, label="checkpoint revision")
        replayed = _require_nonnegative_int(
            self.physical_steps_replayed_on_resume,
            label="physical steps replayed on resume",
        )
        if replayed != 0:
            raise JournalIntegrityError("T10.2.1 forbids physical replay on resume")
        if len({item.lane.lane_id for item in self.lane_reports}) != len(
            self.lane_reports
        ):
            raise JournalIntegrityError("checkpoint contains duplicate lane reports")
        elapsed = float(self.cumulative_active_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("checkpoint active time must be finite and non-negative")
        lane_elapsed = float(self.open_lane_elapsed_seconds)
        if not math.isfinite(lane_elapsed) or lane_elapsed < 0.0:
            raise ValueError("checkpoint lane time must be finite and non-negative")
        registered_lane_ids = {lane.lane_id for lane in source_lane_registry()}
        if self.open_lane_id is None:
            if lane_elapsed != 0.0:
                raise JournalIntegrityError("closed checkpoint retained open-lane time")
        elif self.open_lane_id not in registered_lane_ids:
            raise JournalIntegrityError("checkpoint open lane is not registered")
        expected = canonical_sha256(self.unsigned_dict())
        if self.checkpoint_checksum and self.checkpoint_checksum != expected:
            raise JournalIntegrityError("collection checkpoint checksum drifted")
        object.__setattr__(self, "checkpoint_checksum", expected)

    @property
    def authorized_intent_count(self) -> int:
        return sum(item.issued_intents for item in self.lane_reports)

    @property
    def sealed_event_count(self) -> int:
        return sum(item.sealed_events for item in self.lane_reports)

    @property
    def unresolved_intent_count(self) -> int:
        return sum(item.unresolved_intents for item in self.lane_reports)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "manifest_checksum": self.manifest_checksum,
            "lane_registry_sha256": self.lane_registry_sha256,
            "lane_reports": [item.to_dict() for item in self.lane_reports],
            "cumulative_active_seconds": float(self.cumulative_active_seconds),
            "open_lane_id": self.open_lane_id,
            "open_lane_elapsed_seconds": float(self.open_lane_elapsed_seconds),
            "journal_reconstructed": bool(self.journal_reconstructed),
            "checkpoint_reconstructed": bool(self.checkpoint_reconstructed),
            "physical_steps_replayed_on_resume": int(
                self.physical_steps_replayed_on_resume
            ),
            "revision": int(self.revision),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "checkpoint_checksum": self.checkpoint_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CollectionCheckpoint":
        raw_lanes = payload.get("lane_reports")
        if not isinstance(raw_lanes, list):
            raise JournalIntegrityError("checkpoint lane reports must be a list")
        return cls(
            manifest_checksum=str(payload.get("manifest_checksum", "")),
            lane_registry_sha256=str(payload.get("lane_registry_sha256", "")),
            lane_reports=tuple(
                LaneReport.from_dict(_mapping(item)) for item in raw_lanes
            ),
            cumulative_active_seconds=float(
                payload.get("cumulative_active_seconds", -1.0)
            ),
            open_lane_id=(
                None
                if payload.get("open_lane_id") is None
                else str(payload.get("open_lane_id"))
            ),
            open_lane_elapsed_seconds=float(
                payload.get("open_lane_elapsed_seconds", 0.0)
            ),
            journal_reconstructed=payload.get("journal_reconstructed") is True,
            checkpoint_reconstructed=payload.get("checkpoint_reconstructed") is True,
            physical_steps_replayed_on_resume=int(
                payload.get("physical_steps_replayed_on_resume", -1)
            ),
            revision=int(payload.get("revision", -1)),
            checkpoint_checksum=str(payload.get("checkpoint_checksum", "")),
        )


# Durability primitives must reach the real filesystem even when a test (or
# platform simulation) rebinds the module-level ``os`` to a stub.  Binding a
# private reference at import time keeps write-ahead journalling immune to such
# monkeypatching while ``os.name``/``os.kill`` elsewhere still honour the swap.
_durable_os = os


def _extended_length_path(path: Path) -> Path:
    """Return a Windows extended-length path so deep journals beat ``MAX_PATH``.

    Lane and intent identifiers are 64-character digests, so a nested output
    directory trivially exceeds the legacy 260-character limit.  The ``\\\\?\\``
    prefix lifts that limit and, because every journal path derives from this
    root, it propagates to every ``pathlib``/``os`` operation transparently.
    """

    if _durable_os.name != "nt":
        return path
    absolute = _durable_os.path.abspath(_durable_os.fspath(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC" + absolute[1:])
    return Path("\\\\?\\" + absolute)


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace one same-volume file only after its contents reach durable storage."""

    path = _extended_length_path(path)
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
            handle.write(text)
            handle.flush()
            _durable_os.fsync(handle.fileno())
        _durable_os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability; Windows guarantees same-volume replace."""

    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        # File contents have already been fsynced.  Some filesystems do not
        # expose directory fsync even though same-directory replace is atomic.
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, canonical_json(payload) + "\n")


def _read_canonical_json(path: Path) -> dict[str, Any]:
    path = _extended_length_path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JournalIntegrityError(f"cannot read durable record: {path}") from exc
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise JournalIntegrityError(f"truncated or multiline durable record: {path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JournalIntegrityError(f"invalid durable JSON: {path}") from exc
    if not isinstance(payload, dict) or raw != canonical_json(payload) + "\n":
        raise JournalIntegrityError(f"non-canonical durable JSON: {path}")
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> bool:
    rendered = canonical_json(payload) + "\n"
    path = _extended_length_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def matches_existing() -> bool:
        if not path.exists():
            return False
        existing = _read_canonical_json(path)
        if canonical_json(existing) + "\n" != rendered:
            raise JournalConflictError(f"immutable durable record conflicts: {path}")
        return True

    if matches_existing():
        return False
    lock_path = path.with_name(f".{path.name}.write-once.lock")
    try:
        descriptor = _durable_os.open(
            lock_path,
            _durable_os.O_CREAT | _durable_os.O_EXCL | _durable_os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        # A writer that has already completed the atomic replace is harmless,
        # even if it has not yet removed its claim file.  A claim without a
        # complete target remains an explicit interruption and is never stolen.
        if matches_existing():
            return False
        raise JournalConflictError(
            f"concurrent or interrupted immutable write: {path}"
        ) from exc
    try:
        _durable_os.close(descriptor)
        if matches_existing():
            return False
        _atomic_write_text(path, rendered)
        return True
    finally:
        lock_path.unlink(missing_ok=True)
        _fsync_directory(path.parent)


@dataclass(frozen=True)
class JournalAccounting:
    authorized_intent_count: int
    sealed_event_count: int
    explicitly_unresolved_intent_count: int
    unknown_intent_count: int
    posterior_update_count: int

    @property
    def equation_holds(self) -> bool:
        return self.authorized_intent_count == (
            self.sealed_event_count + self.explicitly_unresolved_intent_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_intent_count": self.authorized_intent_count,
            "sealed_event_count": self.sealed_event_count,
            "explicitly_unresolved_intent_count": (
                self.explicitly_unresolved_intent_count
            ),
            "unknown_intent_count": self.unknown_intent_count,
            "maximum_authorized_intents": SOURCE_MAXIMUM_AUTHORIZED_INTENTS,
            "equation_holds": self.equation_holds,
        }


class DurableCollectionJournal:
    """Per-record atomic journal with deterministic, idempotent keys.

    Immutable intent/event/update records use write-once semantics.  A repeated
    byte-identical write is a no-op; different content at the same logical key
    is a hard provenance failure.  Reset and lane reports are commit markers and
    are likewise immutable once written.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_checksum: str,
    ) -> None:
        self.root = _extended_length_path(Path(root))
        self.manifest_checksum = _require_digest(
            manifest_checksum, label="manifest checksum"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = _signed(
            {
                "format_version": JOURNAL_FORMAT_VERSION,
                "manifest_checksum": self.manifest_checksum,
                "lane_registry_sha256": canonical_sha256(
                    [lane.to_dict() for lane in source_lane_registry()]
                ),
                "maximum_authorized_intents": SOURCE_MAXIMUM_AUTHORIZED_INTENTS,
                "raw_frames_retained": False,
                "raw_action_groundings_retained": False,
            },
            checksum_key="journal_checksum",
        )
        _write_once(self.root / "journal.json", metadata)

    @property
    def checkpoint_path(self) -> Path:
        return self.root.parent / CHECKPOINT_FILENAME

    def _lane_root(self, lane: SourceLaneKey) -> Path:
        return self.root / "lanes" / lane.lane_id

    def _reset_root(self, work: ResetWorkSpec) -> Path:
        return self._lane_root(work.lane) / "resets" / f"{work.reset_index:02d}"

    def _intent_path(self, intent: ActionIntent) -> Path:
        return self._reset_root(
            ResetWorkSpec(
                lane=intent.lane,
                reset_index=intent.reset_index,
                controller=(
                    "balanced_discovery"
                    if intent.lane.split == "discovery"
                    else confirmation_controller_order(intent.lane.seed)[
                        intent.reset_index
                    ]
                ),
            )
        ) / "intents" / f"{intent.step_index:03d}-{intent.intent_id}.json"

    def _event_path(self, intent: ActionIntent) -> Path:
        return self._reset_root(
            ResetWorkSpec(
                lane=intent.lane,
                reset_index=intent.reset_index,
                controller=(
                    "balanced_discovery"
                    if intent.lane.split == "discovery"
                    else confirmation_controller_order(intent.lane.seed)[
                        intent.reset_index
                    ]
                ),
            )
        ) / "events" / f"{intent.step_index:03d}-{intent.intent_id}.json"

    def _update_path(self, intent: ActionIntent) -> Path:
        return self._reset_root(
            ResetWorkSpec(
                lane=intent.lane,
                reset_index=intent.reset_index,
                controller=(
                    "balanced_discovery"
                    if intent.lane.split == "discovery"
                    else confirmation_controller_order(intent.lane.seed)[
                        intent.reset_index
                    ]
                ),
            )
        ) / "posterior_updates" / f"{intent.step_index:03d}-{intent.intent_id}.json"

    def _unresolved_path(self, intent: ActionIntent) -> Path:
        return self._reset_root(
            ResetWorkSpec(
                lane=intent.lane,
                reset_index=intent.reset_index,
                controller=(
                    "balanced_discovery"
                    if intent.lane.split == "discovery"
                    else confirmation_controller_order(intent.lane.seed)[
                        intent.reset_index
                    ]
                ),
            )
        ) / "unresolved" / f"{intent.step_index:03d}-{intent.intent_id}.json"

    def record_intent(self, intent: ActionIntent) -> bool:
        if intent.manifest_checksum != self.manifest_checksum:
            raise JournalIntegrityError("action intent escaped its manifest")
        accounting = self.accounting()
        if (
            not self._intent_path(intent).exists()
            and accounting.authorized_intent_count >= SOURCE_MAXIMUM_AUTHORIZED_INTENTS
        ):
            raise JournalIntegrityError("source action authorization cap exhausted")
        return _write_once(self._intent_path(intent), intent.to_dict())

    def record_physical_event(
        self,
        *,
        intent: ActionIntent,
        event: Mapping[str, Any],
    ) -> PhysicalEventReceipt:
        intent_path = self._intent_path(intent)
        if not intent_path.is_file():
            raise JournalIntegrityError("physical event lacks a durable action intent")
        persisted_intent = ActionIntent.from_dict(_read_canonical_json(intent_path))
        if persisted_intent != intent:
            raise JournalConflictError("physical event intent binding drifted")
        try:
            _t10_2_protocol.verify_event_checksum(event)
        except Exception as exc:
            raise JournalIntegrityError("physical event is not T10.2-sealed") from exc
        event_payload = dict(_jsonable(event))
        receipt = PhysicalEventReceipt(
            intent_id=intent.intent_id,
            intent_checksum=intent.intent_checksum,
            event_id=str(event_payload.get("event_id", "")),
            event_checksum=str(event_payload.get("event_checksum", "")),
            event_payload_sha256=canonical_sha256(event_payload),
        )
        record = {
            "format_version": JOURNAL_FORMAT_VERSION,
            "kind": "sealed_physical_event",
            "intent": intent.to_dict(),
            "receipt": receipt.to_dict(),
            "event": event_payload,
        }
        _write_once(self._event_path(intent), record)
        return receipt

    def record_posterior_update(
        self,
        *,
        intent: ActionIntent,
        receipt: PosteriorUpdateReceipt,
    ) -> bool:
        event_record = self._read_event_record(intent)
        physical = PhysicalEventReceipt.from_dict(
            _mapping(event_record.get("receipt"))
        )
        if (
            receipt.intent_id != intent.intent_id
            or receipt.event_checksum != physical.event_checksum
        ):
            raise JournalConflictError("posterior update escaped its durable event")
        return _write_once(self._update_path(intent), receipt.to_dict())

    def record_unresolved_intent(
        self,
        *,
        intent: ActionIntent,
        reason: Literal[
            "hard_reset_timeout",
            "cooperative_reset_deadline",
            "worker_failed",
            "parent_interrupted",
            "environment_call_unattestable",
        ],
    ) -> UnresolvedIntentReceipt:
        if not self._intent_path(intent).is_file():
            raise JournalIntegrityError("unresolved receipt lacks a durable intent")
        if self._event_path(intent).exists():
            raise JournalConflictError(
                "a sealed physical event cannot also be explicitly unresolved"
            )
        receipt = UnresolvedIntentReceipt(
            intent_id=intent.intent_id,
            intent_checksum=intent.intent_checksum,
            reason=reason,
        )
        _write_once(self._unresolved_path(intent), receipt.to_dict())
        return receipt

    def _read_event_record(self, intent: ActionIntent) -> dict[str, Any]:
        path = self._event_path(intent)
        if not path.is_file():
            raise JournalIntegrityError("posterior update lacks a durable event")
        record = _read_canonical_json(path)
        if record.get("kind") != "sealed_physical_event":
            raise JournalIntegrityError("invalid physical-event journal record")
        persisted_intent = ActionIntent.from_dict(_mapping(record.get("intent")))
        if persisted_intent != intent:
            raise JournalConflictError("event record intent drifted")
        event = _mapping(record.get("event"))
        receipt = PhysicalEventReceipt.from_dict(_mapping(record.get("receipt")))
        if (
            canonical_sha256(event) != receipt.event_payload_sha256
            or str(event.get("event_checksum", "")) != receipt.event_checksum
            or str(event.get("event_id", "")) != receipt.event_id
        ):
            raise JournalIntegrityError("physical event receipt binding drifted")
        try:
            _t10_2_protocol.verify_event_checksum(event)
        except Exception as exc:
            raise JournalIntegrityError("persisted physical event checksum failed") from exc
        return record

    def intents_for_reset(self, work: ResetWorkSpec) -> tuple[ActionIntent, ...]:
        directory = self._reset_root(work) / "intents"
        if not directory.is_dir():
            return ()
        intents = tuple(
            ActionIntent.from_dict(_read_canonical_json(path))
            for path in sorted(directory.glob("*.json"))
        )
        if any(
            intent.lane != work.lane or intent.reset_index != work.reset_index
            for intent in intents
        ):
            raise JournalIntegrityError("action intent escaped its reset")
        if [item.step_index for item in intents] != list(range(len(intents))):
            raise JournalIntegrityError("action intent sequence is not contiguous")
        return intents

    def events_for_reset(self, work: ResetWorkSpec) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for intent in self.intents_for_reset(work):
            path = self._event_path(intent)
            if not path.is_file():
                continue
            record = self._read_event_record(intent)
            events.append(dict(_mapping(record["event"])))
        return tuple(events)

    def updates_for_reset(
        self, work: ResetWorkSpec
    ) -> tuple[PosteriorUpdateReceipt, ...]:
        updates: list[PosteriorUpdateReceipt] = []
        for intent in self.intents_for_reset(work):
            path = self._update_path(intent)
            if not path.is_file():
                continue
            receipt = PosteriorUpdateReceipt.from_dict(_read_canonical_json(path))
            event = self._read_event_record(intent)
            physical = PhysicalEventReceipt.from_dict(_mapping(event["receipt"]))
            if (
                receipt.intent_id != intent.intent_id
                or receipt.event_checksum != physical.event_checksum
            ):
                raise JournalIntegrityError("posterior update receipt binding drifted")
            updates.append(receipt)
        return tuple(updates)

    def reset_accounting(self, work: ResetWorkSpec) -> JournalAccounting:
        intents = self.intents_for_reset(work)
        event_intent_ids: set[str] = set()
        unresolved_intent_ids: set[str] = set()
        unknown = 0
        for intent in intents:
            event_exists = self._event_path(intent).is_file()
            unresolved_exists = self._unresolved_path(intent).is_file()
            if event_exists and unresolved_exists:
                raise JournalIntegrityError(
                    "intent has both a sealed event and an unresolved receipt"
                )
            if event_exists:
                self._read_event_record(intent)
                event_intent_ids.add(intent.intent_id)
            elif unresolved_exists:
                receipt = UnresolvedIntentReceipt.from_dict(
                    _read_canonical_json(self._unresolved_path(intent))
                )
                if (
                    receipt.intent_id != intent.intent_id
                    or receipt.intent_checksum != intent.intent_checksum
                ):
                    raise JournalIntegrityError(
                        "unresolved receipt escaped its action intent"
                    )
                unresolved_intent_ids.add(intent.intent_id)
            else:
                unknown += 1
        expected_intent_paths = {self._intent_path(intent).resolve() for intent in intents}
        expected_event_paths = {self._event_path(intent).resolve() for intent in intents}
        expected_update_paths = {self._update_path(intent).resolve() for intent in intents}
        expected_unresolved_paths = {
            self._unresolved_path(intent).resolve() for intent in intents
        }
        reset_root = self._reset_root(work)
        for directory_name, expected_paths in (
            ("intents", expected_intent_paths),
            ("events", expected_event_paths),
            ("posterior_updates", expected_update_paths),
            ("unresolved", expected_unresolved_paths),
        ):
            directory = reset_root / directory_name
            if directory.is_dir():
                unknown += sum(
                    1
                    for path in directory.iterdir()
                    if not path.is_file() or path.resolve() not in expected_paths
                )
        updates = self.updates_for_reset(work)
        return JournalAccounting(
            authorized_intent_count=len(intents),
            sealed_event_count=len(event_intent_ids),
            explicitly_unresolved_intent_count=len(unresolved_intent_ids),
            unknown_intent_count=unknown,
            posterior_update_count=len(updates),
        )

    def accounting(self) -> JournalAccounting:
        authorized = sealed = unresolved = unknown = updates = 0
        for lane in source_lane_registry():
            for work in reset_work_specs(lane):
                item = self.reset_accounting(work)
                authorized += item.authorized_intent_count
                sealed += item.sealed_event_count
                unresolved += item.explicitly_unresolved_intent_count
                unknown += item.unknown_intent_count
                updates += item.posterior_update_count
        unknown += self._unknown_topology_count()
        result = JournalAccounting(
            authorized_intent_count=authorized,
            sealed_event_count=sealed,
            explicitly_unresolved_intent_count=unresolved,
            unknown_intent_count=unknown,
            posterior_update_count=updates,
        )
        if authorized > SOURCE_MAXIMUM_AUTHORIZED_INTENTS:
            raise JournalIntegrityError("journal exceeds the source action cap")
        return result

    def _unknown_topology_count(self) -> int:
        expected_lanes = {lane.lane_id: lane for lane in source_lane_registry()}
        lanes_root = self.root / "lanes"
        unknown = sum(
            1
            for child in self.root.iterdir()
            if child.name not in {"journal.json", "lanes"}
        )
        if not lanes_root.exists():
            return unknown
        for lane_path in lanes_root.iterdir():
            lane = expected_lanes.get(lane_path.name)
            if lane is None or not lane_path.is_dir():
                unknown += 1
                continue
            expected_resets = {
                f"{work.reset_index:02d}": work for work in reset_work_specs(lane)
            }
            resets_root = lane_path / "resets"
            for child in lane_path.iterdir():
                if child.name not in {"resets", "lane_report.json"}:
                    unknown += 1
            if not resets_root.exists():
                continue
            for reset_path in resets_root.iterdir():
                if reset_path.name not in expected_resets or not reset_path.is_dir():
                    unknown += 1
                    continue
                allowed = {
                    "intents",
                    "events",
                    "posterior_updates",
                    "unresolved",
                    "reset_report.json",
                }
                unknown += sum(
                    1 for child in reset_path.iterdir() if child.name not in allowed
                )
        return unknown

    def write_reset_report(self, report: ResetReport) -> bool:
        accounting = self.reset_accounting(report.work)
        events = self.events_for_reset(report.work)
        if (
            report.issued_intents != accounting.authorized_intent_count
            or report.sealed_events != accounting.sealed_event_count
            or report.unresolved_intents
            != accounting.explicitly_unresolved_intent_count
            or accounting.unknown_intent_count != 0
            or not accounting.equation_holds
            or report.posterior_updates != accounting.posterior_update_count
            or report.event_ids_sha256
            != canonical_sha256([str(event.get("event_id", "")) for event in events])
        ):
            raise JournalIntegrityError("reset report does not reconstruct from journal")
        path = self._reset_root(report.work) / "reset_report.json"
        return _write_once(path, report.to_dict())

    def read_reset_report(self, work: ResetWorkSpec) -> ResetReport | None:
        path = self._reset_root(work) / "reset_report.json"
        if not path.is_file():
            return None
        report = ResetReport.from_dict(_read_canonical_json(path))
        if report.work != work:
            raise JournalIntegrityError("reset report escaped its work specification")
        accounting = self.reset_accounting(work)
        if (
            report.issued_intents != accounting.authorized_intent_count
            or report.sealed_events != accounting.sealed_event_count
            or report.unresolved_intents
            != accounting.explicitly_unresolved_intent_count
            or accounting.unknown_intent_count != 0
            or not accounting.equation_holds
        ):
            raise JournalIntegrityError("persisted reset report accounting drifted")
        return report

    def write_lane_report(self, report: LaneReport) -> bool:
        reconstructed = tuple(
            item
            for work in reset_work_specs(report.lane)
            if (item := self.read_reset_report(work)) is not None
        )
        if report.resets != reconstructed:
            raise JournalIntegrityError("lane report does not reconstruct from reset reports")
        path = self._lane_root(report.lane) / "lane_report.json"
        return _write_once(path, report.to_dict())

    def read_lane_report(self, lane: SourceLaneKey) -> LaneReport | None:
        path = self._lane_root(lane) / "lane_report.json"
        if not path.is_file():
            return None
        report = LaneReport.from_dict(_read_canonical_json(path))
        if report.lane != lane:
            raise JournalIntegrityError("lane report escaped its lane")
        reconstructed = tuple(
            item
            for work in reset_work_specs(lane)
            if (item := self.read_reset_report(work)) is not None
        )
        if report.resets != reconstructed:
            raise JournalIntegrityError("persisted lane report reset binding drifted")
        return report

    def lane_reports(self) -> tuple[LaneReport, ...]:
        return tuple(
            report
            for lane in source_lane_registry()
            if (report := self.read_lane_report(lane)) is not None
        )

    def all_events(self, *, complete_resets_only: bool = True) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for lane in source_lane_registry():
            for work in reset_work_specs(lane):
                report = self.read_reset_report(work)
                if complete_resets_only and (
                    report is None or report.status != "COMPLETE"
                ):
                    continue
                rows.extend(self.events_for_reset(work))
        event_ids = [str(row.get("event_id", "")) for row in rows]
        if not all(event_ids) or len(event_ids) != len(set(event_ids)):
            raise JournalIntegrityError("durable source events have missing/duplicate ids")
        return tuple(rows)

    def write_checkpoint(self, checkpoint: CollectionCheckpoint) -> None:
        if checkpoint.manifest_checksum != self.manifest_checksum:
            raise JournalIntegrityError("checkpoint escaped its manifest")
        expected_registry = canonical_sha256(
            [lane.to_dict() for lane in source_lane_registry()]
        )
        if checkpoint.lane_registry_sha256 != expected_registry:
            raise JournalIntegrityError("checkpoint lane registry drifted")
        reconstructed = self.lane_reports()
        if checkpoint.lane_reports != reconstructed:
            raise JournalIntegrityError("checkpoint lane reports did not reconstruct")
        existing = self.load_checkpoint()
        if existing is not None:
            if checkpoint.to_dict() == existing.to_dict():
                return
            if checkpoint.revision <= existing.revision:
                raise JournalConflictError("checkpoint revision did not advance")
            if len(checkpoint.lane_reports) < len(existing.lane_reports):
                raise JournalConflictError("checkpoint attempted to forget a lane report")
        _atomic_write_json(self.checkpoint_path, checkpoint.to_dict())

    def load_checkpoint(self) -> CollectionCheckpoint | None:
        if not self.checkpoint_path.is_file():
            return None
        checkpoint = CollectionCheckpoint.from_dict(
            _read_canonical_json(self.checkpoint_path)
        )
        if checkpoint.manifest_checksum != self.manifest_checksum:
            raise JournalIntegrityError("checkpoint manifest binding drifted")
        return checkpoint

    def reconstruct_checkpoint(
        self,
        *,
        cumulative_active_seconds: float | None = None,
        open_lane: SourceLaneKey | None = None,
        open_lane_elapsed_seconds: float | None = None,
        close_open_lane: bool = False,
    ) -> CollectionCheckpoint:
        lanes = self.lane_reports()
        reset_reports = _all_reset_reports(self)
        reset_active_seconds = sum(item.elapsed_seconds for item in reset_reports)
        existing = self.load_checkpoint()
        if cumulative_active_seconds is None:
            observed_active_seconds = max(
                reset_active_seconds,
                (
                    existing.cumulative_active_seconds
                    if existing is not None
                    else 0.0
                ),
            )
        else:
            observed_active_seconds = float(cumulative_active_seconds)
            if (
                not math.isfinite(observed_active_seconds)
                or observed_active_seconds < 0.0
            ):
                raise JournalIntegrityError(
                    "checkpoint cumulative active duration is invalid"
                )
            observed_active_seconds = max(
                observed_active_seconds,
                reset_active_seconds,
                (
                    existing.cumulative_active_seconds
                    if existing is not None
                    else 0.0
                ),
            )
        if close_open_lane and open_lane is not None:
            raise ValueError("checkpoint cannot open and close a lane together")
        if close_open_lane:
            observed_open_lane_id = None
            observed_open_lane_seconds = 0.0
        elif open_lane is not None:
            if open_lane_elapsed_seconds is None:
                raise ValueError("open-lane checkpoint requires its active duration")
            observed_open_lane_id = open_lane.lane_id
            observed_open_lane_seconds = float(open_lane_elapsed_seconds)
            if (
                not math.isfinite(observed_open_lane_seconds)
                or observed_open_lane_seconds < 0.0
            ):
                raise JournalIntegrityError("checkpoint open-lane duration is invalid")
            if existing is not None and existing.open_lane_id == observed_open_lane_id:
                observed_open_lane_seconds = max(
                    observed_open_lane_seconds,
                    existing.open_lane_elapsed_seconds,
                )
        elif existing is not None:
            observed_open_lane_id = existing.open_lane_id
            observed_open_lane_seconds = existing.open_lane_elapsed_seconds
        else:
            observed_open_lane_id = None
            observed_open_lane_seconds = 0.0
        accounting = self.accounting()
        journal_metadata = _read_canonical_json(self.root / "journal.json")
        _verify_signed(journal_metadata, checksum_key="journal_checksum")
        expected_registry = canonical_sha256(
            [lane.to_dict() for lane in source_lane_registry()]
        )
        journal_reconstructed = bool(
            journal_metadata.get("manifest_checksum") == self.manifest_checksum
            and journal_metadata.get("lane_registry_sha256") == expected_registry
            and accounting.authorized_intent_count
            <= SOURCE_MAXIMUM_AUTHORIZED_INTENTS
        )
        existing_matches = bool(
            existing is not None
            and existing.manifest_checksum == self.manifest_checksum
            and existing.lane_registry_sha256 == expected_registry
            and existing.lane_reports == lanes
            and existing.cumulative_active_seconds == observed_active_seconds
            and existing.open_lane_id == observed_open_lane_id
            and existing.open_lane_elapsed_seconds == observed_open_lane_seconds
            and existing.journal_reconstructed == journal_reconstructed
            and existing.checkpoint_reconstructed
            and existing.physical_steps_replayed_on_resume == 0
        )
        if existing_matches and existing is not None:
            return existing
        revision = 0 if existing is None else existing.revision + 1
        checkpoint = CollectionCheckpoint(
            manifest_checksum=self.manifest_checksum,
            lane_registry_sha256=expected_registry,
            lane_reports=lanes,
            cumulative_active_seconds=observed_active_seconds,
            open_lane_id=observed_open_lane_id,
            open_lane_elapsed_seconds=observed_open_lane_seconds,
            journal_reconstructed=journal_reconstructed,
            checkpoint_reconstructed=True,
            physical_steps_replayed_on_resume=0,
            revision=revision,
        )
        self.write_checkpoint(checkpoint)
        reloaded = self.load_checkpoint()
        if reloaded != checkpoint:
            raise JournalIntegrityError("checkpoint failed canonical reconstruction")
        return checkpoint

    def assert_safe_resume_boundary(self, work: ResetWorkSpec) -> None:
        report = self.read_reset_report(work)
        if report is not None:
            if report.status == "COMPLETE":
                return
            raise ResumeRefusalError(
                f"reset {work.work_id} already ended as {report.status}"
            )
        accounting = self.reset_accounting(work)
        if accounting.authorized_intent_count:
            raise ResumeRefusalError(
                "partial physical reset has no safe commit marker; replay is forbidden"
            )


@dataclass(frozen=True)
class WorkerOutcome:
    status: Literal["COMPLETE", "COOPERATIVE_STOP", "HARD_TIMEOUT", "FAILED"]
    elapsed_seconds: float
    payload: Mapping[str, Any]
    error_kind: str | None = None


class ResetWatchdog(Protocol):
    """Injectable parent-side supervisor for one spawned reset process."""

    def supervise(
        self,
        process: Any,
        *,
        work: ResetWorkSpec,
        cancel_event: Any,
        outbound_queue: Any,
        inbound_queue: Any,
        message_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        cooperative_seconds: float,
        hard_seconds: float,
        started_at: float,
    ) -> WorkerOutcome: ...


class ProcessResetWatchdog:
    """Default hard watchdog; only the parent owns timeout authority."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        poll_seconds: float = 0.05,
        termination_grace_seconds: float = 2.0,
    ) -> None:
        self._clock = clock
        self._poll_seconds = float(poll_seconds)
        self._termination_grace_seconds = float(termination_grace_seconds)
        if self._poll_seconds <= 0.0 or self._termination_grace_seconds < 0.0:
            raise ValueError("watchdog polling/grace intervals are invalid")

    def supervise(
        self,
        process: Any,
        *,
        work: ResetWorkSpec,
        cancel_event: Any,
        outbound_queue: Any,
        inbound_queue: Any,
        message_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        cooperative_seconds: float,
        hard_seconds: float,
        started_at: float,
    ) -> WorkerOutcome:
        started = float(started_at)
        if not math.isfinite(started):
            self._terminate(process)
            raise WorkerProtocolError("watchdog reset start is invalid")
        cooperative_requested = False
        terminal_payload: Mapping[str, Any] | None = None
        terminal_error: str | None = None
        while True:
            now = float(self._clock())
            elapsed = now - started
            if not math.isfinite(elapsed) or elapsed < 0.0:
                self._terminate(process)
                raise WorkerProtocolError("watchdog monotonic clock regressed")
            if elapsed >= hard_seconds:
                if not self._terminate(process):
                    raise WorkerTimeoutError(
                        "reset worker remained alive after verified hard termination"
                    )
                return WorkerOutcome(
                    status="HARD_TIMEOUT",
                    elapsed_seconds=elapsed,
                    payload={
                        "work_id": work.work_id,
                        "stop_reason": "hard_reset_timeout",
                    },
                    error_kind="WorkerTimeoutError",
                )
            if not cooperative_requested and elapsed >= cooperative_seconds:
                cancel_event.set()
                cooperative_requested = True
            timeout = min(self._poll_seconds, max(0.001, hard_seconds - elapsed))
            try:
                raw = outbound_queue.get(timeout=timeout)
            except queue.Empty:
                raw = None
            if raw is not None:
                if not isinstance(raw, Mapping):
                    self._terminate(process)
                    raise WorkerProtocolError("reset worker emitted a non-mapping message")
                message = dict(raw)
                kind = str(message.get("kind", ""))
                if message.get("work_id") != work.work_id:
                    self._terminate(process)
                    raise WorkerProtocolError("reset worker escaped its work id")
                if kind == "reset_finished":
                    terminal_payload = _mapping(message.get("payload", {}))
                elif kind == "worker_failed":
                    terminal_payload = _mapping(message.get("payload", {}))
                    terminal_error = str(message.get("error_kind", "WorkerProtocolError"))
                else:
                    try:
                        acknowledgement = message_handler(message)
                    except WorkerTimeoutError:
                        if not self._terminate(process):
                            raise WorkerTimeoutError(
                                "reset worker remained alive after intent refusal"
                            )
                        elapsed = max(0.0, float(self._clock()) - started)
                        return WorkerOutcome(
                            status="COOPERATIVE_STOP",
                            elapsed_seconds=elapsed,
                            payload={
                                "completed": False,
                                "stop_reason": "cooperative_reset_deadline",
                            },
                            error_kind="WorkerTimeoutError",
                        )
                    except Exception:
                        self._terminate(process)
                        raise
                    if acknowledgement is not None:
                        inbound_queue.put(dict(acknowledgement))
            if terminal_payload is not None and not process.is_alive():
                process.join(timeout=self._termination_grace_seconds)
                elapsed = max(0.0, float(self._clock()) - started)
                if terminal_error is not None:
                    return WorkerOutcome(
                        status="FAILED",
                        elapsed_seconds=elapsed,
                        payload=terminal_payload,
                        error_kind=terminal_error,
                    )
                completed = terminal_payload.get("completed") is True
                return WorkerOutcome(
                    status=(
                        "COMPLETE"
                        if completed
                        else "COOPERATIVE_STOP"
                        if cooperative_requested
                        else "FAILED"
                    ),
                    elapsed_seconds=elapsed,
                    payload=terminal_payload,
                    error_kind=None if completed else "ResetIncomplete",
                )
            if not process.is_alive():
                # Drain messages queued immediately before process exit.
                process.join(timeout=self._termination_grace_seconds)
                try:
                    while True:
                        raw = outbound_queue.get_nowait()
                        if isinstance(raw, Mapping) and raw.get("work_id") == work.work_id:
                            if raw.get("kind") == "reset_finished":
                                terminal_payload = _mapping(raw.get("payload", {}))
                            elif raw.get("kind") == "worker_failed":
                                terminal_payload = _mapping(raw.get("payload", {}))
                                terminal_error = str(
                                    raw.get("error_kind", "WorkerProtocolError")
                                )
                            else:
                                acknowledgement = message_handler(raw)
                                if acknowledgement is not None:
                                    inbound_queue.put(dict(acknowledgement))
                except queue.Empty:
                    pass
                elapsed = max(0.0, float(self._clock()) - started)
                if terminal_payload is not None and terminal_error is None:
                    completed = terminal_payload.get("completed") is True
                    return WorkerOutcome(
                        status="COMPLETE" if completed else "FAILED",
                        elapsed_seconds=elapsed,
                        payload=terminal_payload,
                        error_kind=None if completed else "ResetIncomplete",
                    )
                return WorkerOutcome(
                    status="FAILED",
                    elapsed_seconds=elapsed,
                    payload=terminal_payload or {"stop_reason": "worker_exited"},
                    error_kind=terminal_error or "WorkerExitedWithoutResult",
                )

    def _terminate(self, process: Any) -> bool:
        if os.name == "nt" and process.is_alive():
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and pid > 0:
                try:
                    completed = subprocess.run(
                        [
                            "taskkill.exe",
                            "/PID",
                            str(pid),
                            "/T",
                            "/F",
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(
                            subprocess, "CREATE_NO_WINDOW", 0
                        ),
                        timeout=self._termination_grace_seconds,
                    )
                    if getattr(completed, "returncode", 1) != 0:
                        # Fall through to the process-handle termination below.
                        pass
                except (OSError, subprocess.TimeoutExpired):
                    pass
                process.join(timeout=self._termination_grace_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self._termination_grace_seconds)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
                process.join(timeout=self._termination_grace_seconds)
        return not process.is_alive()


def _worker_message(
    outbound_queue: Any,
    work: ResetWorkSpec,
    *,
    kind: str,
    payload: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    outbound_queue.put(
        {
            "kind": kind,
            "work_id": work.work_id,
            "payload": {} if payload is None else dict(_jsonable(payload)),
            **_jsonable(extra),
        }
    )


def _await_parent_ack(
    inbound_queue: Any,
    *,
    expected_kind: str,
    expected_id: str,
) -> Mapping[str, Any]:
    while True:
        raw = inbound_queue.get()
        if not isinstance(raw, Mapping):
            raise WorkerProtocolError("parent acknowledgement is not a mapping")
        if raw.get("kind") != expected_kind or raw.get("record_id") != expected_id:
            raise WorkerProtocolError("parent acknowledgement order drifted")
        if raw.get("accepted") is not True:
            raise WorkerProtocolError(
                str(raw.get("reason", "parent refused durable record"))
            )
        return raw


def _posterior_fingerprint(posterior: Any | None) -> str | None:
    if posterior is None:
        return None
    return str(_science._posterior_mass_fingerprint(posterior))


def _grounding_commitment(action: Any) -> str:
    return canonical_sha256({"grounding": _science._grounding_key(action)})


def _counter_from_serialized(
    raw: Any,
    *,
    schema_keys: bool,
) -> Counter[Any]:
    if not isinstance(raw, Mapping):
        return Counter()
    counter: Counter[Any] = Counter()
    for key, value in raw.items():
        count = _require_nonnegative_int(value, label="continuation count")
        counter[_schema_count_from_key(str(key)) if schema_keys else str(key)] = count
    return counter


def _serialize_schema_counter(counter: Counter[tuple[str, int]]) -> dict[str, int]:
    return {
        _schema_count_key(name, arity): int(count)
        for (name, arity), count in sorted(counter.items())
    }


def _rank_actions_durable(
    science_environment: Any,
    legal: Sequence[Any],
    *,
    controller: str,
    donor_prior: Counter[tuple[str, int]],
    posterior_scores: Mapping[tuple[str, int], float],
    independent_counts: Counter[tuple[str, int]],
    learned_counts: Counter[tuple[str, int]],
    grounding_counts: Counter[str],
    capacity_slots: int,
    reset_index: int,
    step_index: int,
) -> list[Any]:
    if controller != "balanced_discovery":
        return science_environment._rank_actions(
            legal,
            controller=controller,
            donor_prior=donor_prior,
            posterior_scores=posterior_scores,
            independent_counts=independent_counts,
            learned_counts=learned_counts,
            capacity_slots=capacity_slots,
            grounding_counts=Counter(),
            reset_index=reset_index,
            step_index=step_index,
        )

    def rank(action: Any) -> tuple[Any, ...]:
        schema = _science._action_schema(action)
        grounding = _grounding_commitment(action)
        tie = _science._stable_hash(
            {
                "seed": science_environment.seed,
                "reset": reset_index,
                "step": step_index,
                "grounding": _science._grounding_key(action),
            }
        )
        return (
            science_environment.factory._discovery_counts[
                science_environment.game_id
            ][schema],
            grounding_counts[grounding],
            tie,
        )

    return sorted(legal, key=rank)


class _ParentJournalMessageHandler:
    def __init__(
        self,
        *,
        journal: DurableCollectionJournal,
        work: ResetWorkSpec,
        manifest: Mapping[str, Any],
        clock: Callable[[], float],
        intent_deadline: float,
    ) -> None:
        self.journal = journal
        self.work = work
        self.manifest = manifest
        self._clock = clock
        self._intent_deadline = float(intent_deadline)
        self._intents: dict[str, ActionIntent] = {
            intent.intent_id: intent for intent in journal.intents_for_reset(work)
        }

    def __call__(self, message: Mapping[str, Any]) -> Mapping[str, Any] | None:
        kind = str(message.get("kind", ""))
        payload = _mapping(message.get("payload", {}))
        if kind in {"reset_started", "reset_operation"}:
            return None
        if kind == "action_intent":
            now = float(self._clock())
            if not math.isfinite(now) or now >= self._intent_deadline:
                raise WorkerTimeoutError(
                    "cooperative deadline reached before intent authorization"
                )
            intent = ActionIntent.from_dict(payload)
            if intent.lane != self.work.lane or intent.reset_index != self.work.reset_index:
                raise WorkerProtocolError("worker intent escaped its reset")
            self.journal.record_intent(intent)
            persisted_at = float(self._clock())
            if (
                not math.isfinite(persisted_at)
                or persisted_at >= self._intent_deadline
            ):
                self.journal.record_unresolved_intent(
                    intent=intent,
                    reason="cooperative_reset_deadline",
                )
                raise WorkerTimeoutError(
                    "cooperative deadline crossed while persisting intent"
                )
            self._intents[intent.intent_id] = intent
            return {
                "kind": "intent_ack",
                "record_id": intent.intent_id,
                "accepted": True,
            }
        if kind == "physical_event":
            intent_id = str(message.get("intent_id", ""))
            intent = self._intents.get(intent_id)
            if intent is None:
                raise WorkerProtocolError("worker event lacks an acknowledged intent")
            raw_event = dict(_mapping(payload.get("event")))
            sealed = _t10_2_protocol._fresh_source_event(
                raw_event,
                game_id=self.work.lane.game_id,
                seed=self.work.lane.seed,
                split=self.work.lane.split,
                manifest=self.manifest,
            )
            receipt = self.journal.record_physical_event(intent=intent, event=sealed)
            return {
                "kind": "event_ack",
                "record_id": intent.intent_id,
                "accepted": True,
                "event_checksum": receipt.event_checksum,
                "sealed_event": sealed,
            }
        if kind == "posterior_update":
            intent_id = str(message.get("intent_id", ""))
            intent = self._intents.get(intent_id)
            if intent is None:
                raise WorkerProtocolError("posterior receipt lacks an intent")
            receipt = PosteriorUpdateReceipt.from_dict(payload)
            self.journal.record_posterior_update(intent=intent, receipt=receipt)
            return {
                "kind": "posterior_ack",
                "record_id": intent.intent_id,
                "accepted": True,
            }
        raise WorkerProtocolError(f"unregistered worker message: {kind}")


class T10_2_1SourceFactory(_science.T10_2SourceFactory):
    """Spawn-safe source factory retaining the frozen T10.2 scientific core."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any] | None = None,
        runtime_loader: Callable[[], Any] | None = None,
        bundle_builder: Callable[..., Any] | None = None,
        watchdog: ResetWatchdog | None = None,
    ) -> None:
        super().__init__(
            manifest=manifest,
            runtime_loader=runtime_loader,
            bundle_builder=bundle_builder,
        )
        self.manifest = dict(manifest or {})
        self.watchdog = watchdog

    def __getstate__(self) -> dict[str, Any]:
        if self._runtime is not None:
            raise WorkerProtocolError("a loaded runtime cannot cross the spawn boundary")
        state = dict(self.__dict__)
        # A parent-side watchdog can own locks or test clocks and is never used
        # inside the child process.
        state["watchdog"] = None
        return state

    def restore_completed_discovery(
        self, events: Sequence[Mapping[str, Any]]
    ) -> None:
        self._discovery_events = {game: [] for game in SOURCE_GAMES}
        self._discovery_counts = {game: Counter() for game in SOURCE_GAMES}
        seen: set[str] = set()
        for raw in events:
            event = dict(raw)
            try:
                _t10_2_protocol.verify_event_checksum(event)
            except Exception as exc:
                raise JournalIntegrityError(
                    "discovery hydration received an unsealed physical event"
                ) from exc
            if event.get("split") != "discovery":
                raise JournalIntegrityError("confirmation evidence entered discovery hydration")
            game_id = str(event.get("game_id", ""))
            seed = int(event.get("seed", -1))
            event_id = str(event.get("event_id", ""))
            if game_id not in SOURCE_GAMES or seed not in DISCOVERY_SEEDS or not event_id:
                raise JournalIntegrityError("discovery hydration provenance drifted")
            if event_id in seen:
                raise JournalIntegrityError("duplicate discovery hydration event")
            seen.add(event_id)
            self._remember_discovery(game_id, event)

    def clone_for_worker(self) -> "T10_2_1SourceFactory":
        return T10_2_1SourceFactory(
            manifest=self.manifest,
            runtime_loader=self._runtime_loader,
            bundle_builder=self._bundle_builder,
        )

    def run_reset(
        self,
        *,
        work: ResetWorkSpec,
        journal: DurableCollectionJournal,
        discovery_events: Sequence[Mapping[str, Any]],
        continuation: Mapping[str, Any],
        process_context: Any,
        lane_remaining_seconds: float,
        cooperative_collection_remaining_seconds: float,
        absolute_collection_remaining_seconds: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> WorkerOutcome:
        journal.assert_safe_resume_boundary(work)
        cooperative_seconds = min(
            work.cooperative_seconds,
            float(lane_remaining_seconds),
            float(cooperative_collection_remaining_seconds),
        )
        hard_seconds = min(
            work.hard_seconds,
            float(lane_remaining_seconds),
            float(absolute_collection_remaining_seconds),
        )
        if cooperative_seconds <= 0.0 or hard_seconds <= 0.0:
            global_budget_exhausted = bool(
                float(cooperative_collection_remaining_seconds) <= 0.0
                or float(absolute_collection_remaining_seconds) <= 0.0
            )
            return WorkerOutcome(
                status="COOPERATIVE_STOP",
                elapsed_seconds=0.0,
                payload={
                    "completed": False,
                    "stop_reason": (
                        "registered_collection_deadline"
                        if global_budget_exhausted
                        else "cooperative_reset_deadline"
                    ),
                },
                error_kind="WorkerTimeoutError",
            )
        reset_started = float(clock())
        if not math.isfinite(reset_started):
            raise WorkerProtocolError("reset monotonic start is invalid")
        hard_deadline_monotonic = time.monotonic() + hard_seconds
        context = process_context or multiprocessing.get_context("spawn")
        outbound_queue = context.Queue()
        inbound_queue = context.Queue()
        cancel_event = context.Event()
        process = context.Process(
            target=_reset_worker_entry,
            args=(
                self.clone_for_worker(),
                work.to_dict(),
                tuple(dict(item) for item in discovery_events),
                dict(_jsonable(continuation)),
                cancel_event,
                outbound_queue,
                inbound_queue,
            ),
            name=f"sage-t10-2-1-{work.work_id[:12]}",
        )
        # The kill switch is live before Process.start().  It therefore bounds
        # spawn itself as well as journal fsyncs, ACK queue operations, joins,
        # and the scientific worker.  Killing the collector tree is deliberate:
        # a parent blocked in an ACK cannot safely report a reset-level timeout.
        hard_cancel_event = threading.Event()
        hard_guard = threading.Thread(
            target=_reset_hard_watchdog_entry,
            args=(
                os.getpid(),
                hard_deadline_monotonic,
                hard_cancel_event,
            ),
            name=f"sage-t10-2-1-reset-hard-watchdog-{work.work_id[:8]}",
            daemon=True,
        )
        try:
            hard_guard.start()
        except Exception as exc:
            raise WorkerProtocolError(
                "external reset hard watchdog could not start"
            ) from exc
        try:
            process.start()
            worker_pid = getattr(process, "pid", None)
            if not isinstance(worker_pid, int) or worker_pid <= 0:
                ProcessResetWatchdog(clock=clock)._terminate(process)
                raise WorkerProtocolError("spawned reset worker lacks a process id")
            handler = _ParentJournalMessageHandler(
                journal=journal,
                work=work,
                manifest=self.manifest,
                clock=clock,
                intent_deadline=(
                    reset_started + min(cooperative_seconds, hard_seconds)
                ),
            )
            watchdog = self.watchdog or ProcessResetWatchdog(clock=clock)
            return watchdog.supervise(
                process,
                work=work,
                cancel_event=cancel_event,
                outbound_queue=outbound_queue,
                inbound_queue=inbound_queue,
                message_handler=handler,
                cooperative_seconds=cooperative_seconds,
                hard_seconds=hard_seconds,
                started_at=reset_started,
            )
        finally:
            hard_cancel_event.set()
            hard_guard.join(timeout=2.0)


def _reset_worker_entry(
    factory: T10_2_1SourceFactory,
    work_payload: Mapping[str, Any],
    discovery_events: Sequence[Mapping[str, Any]],
    continuation: Mapping[str, Any],
    cancel_event: Any,
    outbound_queue: Any,
    inbound_queue: Any,
) -> None:
    work: ResetWorkSpec | None = None
    try:
        work = ResetWorkSpec.from_dict(work_payload)
        payload = _collect_reset_scientific(
            factory=factory,
            work=work,
            discovery_events=discovery_events,
            continuation=continuation,
            cancel_event=cancel_event,
            outbound_queue=outbound_queue,
            inbound_queue=inbound_queue,
        )
        _worker_message(
            outbound_queue,
            work,
            kind="reset_finished",
            payload=payload,
        )
    except BaseException as exc:  # noqa: BLE001 - child must attest every exit.
        if work is None:
            # The parent still requires the deterministic work id to associate
            # a spawn/deserialization failure with the correct reset.
            try:
                work = ResetWorkSpec.from_dict(work_payload)
            except Exception:
                return
        _worker_message(
            outbound_queue,
            work,
            kind="worker_failed",
            payload={"completed": False, "stop_reason": "worker_exception"},
            error_kind=type(exc).__name__,
        )


def _operation(
    outbound_queue: Any,
    work: ResetWorkSpec,
    name: str,
    function: Callable[[], Any],
) -> Any:
    _worker_message(
        outbound_queue,
        work,
        kind="reset_operation",
        payload={"operation": name, "stage": "started"},
    )
    result = function()
    _worker_message(
        outbound_queue,
        work,
        kind="reset_operation",
        payload={"operation": name, "stage": "finished"},
    )
    return result


def _collect_reset_scientific(
    *,
    factory: T10_2_1SourceFactory,
    work: ResetWorkSpec,
    discovery_events: Sequence[Mapping[str, Any]],
    continuation: Mapping[str, Any],
    cancel_event: Any,
    outbound_queue: Any,
    inbound_queue: Any,
) -> dict[str, Any]:
    """Run one physical reset while preserving the frozen T10.2 policy."""

    factory.restore_completed_discovery(discovery_events)
    lane = work.lane
    training_games = tuple(game for game in SOURCE_GAMES if game != lane.game_id)
    science_environment = _science._SourceLaneEnvironment(
        factory=factory,
        game_id=lane.game_id,
        seed=lane.seed,
        split=lane.split,
        held_out_game=(
            None if lane.split == "discovery" else lane.game_id
        ),
        training_games=training_games,
    )
    runtime = factory._load_runtime()
    environment: Any | None = None
    events: list[dict[str, Any]] = []
    reset_error_count = 0
    reset_online_observations = 0
    stop_reason: str | None = None
    completed = True

    donor_prior: Counter[tuple[str, int]] = Counter()
    donor_posterior: Any | None = None
    donor_candidates: tuple[JointGaugeHypothesis, ...] = ()
    donor_events: tuple[Mapping[str, Any], ...] = ()
    posterior_scores: dict[tuple[str, int], float] = {}
    posterior_metadata: dict[str, Any] = {}
    independent_candidates: (
        FactorizedCandidateBank | tuple[JointGaugeHypothesis, ...]
    ) = ()
    independent_refusal = ""
    if lane.split != "discovery":
        if any(not factory._discovery_events[game] for game in training_games):
            raise ResumeRefusalError(
                "confirmation reset lacks completed discovery donors"
            )
        donor_prior = factory._donor_prior(lane.game_id, training_games)
        (
            donor_posterior,
            donor_candidates,
            donor_events,
            posterior_scores,
            posterior_metadata,
        ) = factory._donor_posterior_scores(training_games)
        try:
            independent_candidates = _science._capacity_matched_independent_bank(
                donor_candidates
            )
        except _t10_2_protocol.DataGateError as exc:
            independent_refusal = str(exc)

    learned_reset_count = int(continuation.get("learned_reset_count", 0))
    online_held_out_observations = int(
        continuation.get("online_held_out_observations", 0)
    )
    independent_observations = int(
        continuation.get("independent_observations", 0)
    )
    independent_counts = _counter_from_serialized(
        continuation.get("independent_schema_counts", {}), schema_keys=True
    )
    learned_counts = _counter_from_serialized(
        continuation.get("learned_schema_counts", {}), schema_keys=True
    )
    grounding_counts = _counter_from_serialized(
        continuation.get("grounding_counts", {}), schema_keys=False
    )
    active_posterior: Any | None = None
    if work.controller == "learned":
        if learned_reset_count == 0:
            active_posterior = donor_posterior
            reset_error_count += int(posterior_metadata.get("posterior_errors", 0))
        else:
            (
                active_posterior,
                _bank,
                _observations,
                fit_errors,
            ) = _science._fit_compact_posterior(
                donor_events,
                candidates=donor_candidates,
                maximum_candidates=256,
            )
            reset_error_count += len(fit_errors)
        learned_reset_count += 1
    elif work.controller == "capacity_matched_independent":
        if isinstance(independent_candidates, FactorizedCandidateBank):
            (
                active_posterior,
                _bank,
                _observations,
                fit_errors,
            ) = _science._fit_compact_posterior(
                donor_events,
                candidates=independent_candidates,
                posterior_factory=FactorizedGaugeProgramPosterior,
                maximum_candidates=256,
            )
            reset_error_count += len(fit_errors)

    capacity_slots = int(
        posterior_metadata.get("posterior_candidates", len(donor_prior))
    )
    initial_particles, initial_classes = _science._effective_posterior_capacity(
        active_posterior
    )
    controller_frame_states: dict[str, Any] = {}
    sequence_prefix: list[Any] = []

    try:
        environment = _operation(
            outbound_queue,
            work,
            "open",
            lambda: _science._open_runtime(runtime, lane.game_id, lane.seed),
        )
        science_environment._environment = environment
        frame = _operation(
            outbound_queue,
            work,
            "reset",
            lambda: _science._reset_runtime(runtime, environment),
        )
        before = _operation(
            outbound_queue,
            work,
            "snapshot_before",
            lambda: _science._snapshot_runtime(runtime, frame),
        )
        _worker_message(
            outbound_queue,
            work,
            kind="reset_started",
            payload={"controller": work.controller},
        )
        for step_index in range(work.maximum_actions):
            if cancel_event.is_set():
                stop_reason = "cooperative_reset_deadline"
                completed = False
                break
            if _science._is_terminal(before):
                stop_reason = (
                    "game_over" if _science._is_game_over(before) else "terminal"
                )
                break
            legal = _operation(
                outbound_queue,
                work,
                "legal_actions",
                lambda: _science._legal_runtime(runtime, environment),
            )
            legal = tuple(action for action in legal if _science._action_name(action))
            if not legal:
                stop_reason = "no_legal_actions"
                break
            if active_posterior is not None:
                posterior_scores = _science._posterior_action_scores(active_posterior)
            ranked = _rank_actions_durable(
                science_environment,
                legal,
                controller=work.controller,
                donor_prior=donor_prior,
                posterior_scores=posterior_scores,
                independent_counts=independent_counts,
                learned_counts=learned_counts,
                grounding_counts=grounding_counts,
                capacity_slots=capacity_slots,
                reset_index=work.reset_index,
                step_index=step_index,
            )
            decision_metadata = {
                "engine_used": False,
                "reason": "ranked_nonlearned_controller",
                "normalized_entropy": None,
            }
            selected = ranked[0]
            if active_posterior is not None:
                selected, decision_metadata = science_environment._posterior_decision(
                    legal,
                    posterior=active_posterior,
                    frame_states=controller_frame_states,
                    fallback=selected,
                )
                if selected is None:
                    stop_reason = "policy_abstained"
                    break

            if cancel_event.is_set():
                stop_reason = "cooperative_reset_deadline"
                completed = False
                break
            schema = _science._action_schema(selected)
            grounding = _grounding_commitment(selected)
            grounding_counts[grounding] += 1
            if work.controller == "capacity_matched_independent":
                independent_counts[schema] += 1
            elif work.controller == "learned":
                learned_counts[schema] += 1
            event_id = _science._stable_hash(
                {
                    "runtime": _science.RUNTIME_FORMAT_VERSION,
                    "lane": [lane.game_id, lane.seed, lane.split],
                    "reset": work.reset_index,
                    "step": step_index,
                    "controller": work.controller,
                }
            )
            intent = ActionIntent(
                lane=lane,
                reset_index=work.reset_index,
                step_index=step_index,
                action=selected,
                manifest_checksum=factory.manifest_checksum,
            )
            _worker_message(
                outbound_queue,
                work,
                kind="action_intent",
                payload=intent.to_dict(),
            )
            _await_parent_ack(
                inbound_queue,
                expected_kind="intent_ack",
                expected_id=intent.intent_id,
            )

            after_frame = _operation(
                outbound_queue,
                work,
                "step",
                lambda: _science._step_runtime(runtime, environment, selected),
            )
            after = _operation(
                outbound_queue,
                work,
                "snapshot_after",
                lambda: _science._snapshot_runtime(
                    runtime,
                    after_frame,
                    fallback_available_actions=legal,
                ),
            )
            bundle = _science._make_bundle(
                factory._bundle_builder,
                before=before,
                after=after,
                action=selected,
                legal_actions=legal,
                event_id=event_id,
                step_index=step_index,
                game_id=lane.game_id,
            )

            preview_error = ""
            preview_posterior = None
            next_online_held_out = online_held_out_observations
            next_independent = independent_observations
            next_frame_states = controller_frame_states
            if active_posterior is not None:
                try:
                    preview_posterior = copy.deepcopy(active_posterior)
                    preview_posterior.observe(bundle)
                    if work.controller == "learned":
                        next_online_held_out += 1
                    else:
                        next_independent += 1
                except Exception as exc:  # noqa: BLE001 - encoded, then compared.
                    preview_error = type(exc).__name__
                    reset_error_count += 1
                    preview_posterior = None
                next_frame_states = {
                    projection.frame_id: projection.after.state
                    for projection in bundle.projections
                }
            next_sequence_prefix = [
                *sequence_prefix,
                (
                    bundle.action,
                    tuple(bundle.events),
                    {
                        projection.frame_id: projection.before.state
                        for projection in bundle.projections
                    },
                ),
            ]
            sequence_ranking = None
            if (
                work.controller == "learned"
                and preview_posterior is not None
                and not preview_error
            ):
                sequence_ranking = rank_option_sequence_signatures(
                    preview_posterior,
                    next_sequence_prefix,
                )
            progressing_sequence_rank = (
                sequence_ranking.best_compatible_rank
                if sequence_ranking is not None
                and _science._event_labels(bundle)["progress"]
                else None
            )
            event = _science._compact_event(
                bundle,
                controller=work.controller,
                reset_index=work.reset_index,
                step_index=step_index,
                progressing_sequence_rank=progressing_sequence_rank,
                donor_game_count=len(training_games),
                capacity_slots=capacity_slots,
            )
            event["selection"].update(
                _selection_metadata(
                    controller=work.controller,
                    posterior_metadata=posterior_metadata,
                    independent_candidates=independent_candidates,
                    independent_refusal=independent_refusal,
                    sequence_ranking=sequence_ranking,
                    decision_metadata=decision_metadata,
                    online_held_out_observations=next_online_held_out,
                    independent_observations=next_independent,
                    active_posterior=active_posterior,
                )
            )
            event["selection"]["posterior_update_error"] = preview_error
            _science._assert_compact_event_budget(event)
            _worker_message(
                outbound_queue,
                work,
                kind="physical_event",
                payload={"event": event},
                intent_id=intent.intent_id,
            )
            event_ack = _await_parent_ack(
                inbound_queue,
                expected_kind="event_ack",
                expected_id=intent.intent_id,
            )
            sealed_event = dict(_mapping(event_ack.get("sealed_event")))
            event_checksum = str(event_ack.get("event_checksum", ""))

            actual_error = ""
            update_status: Literal["APPLIED", "SKIPPED", "REFUSED"] = "SKIPPED"
            if active_posterior is not None:
                try:
                    active_posterior.observe(bundle)
                    update_status = "APPLIED"
                except Exception as exc:  # noqa: BLE001 - audited refusal.
                    actual_error = type(exc).__name__
                    update_status = "REFUSED"
                if actual_error != preview_error:
                    raise WorkerProtocolError(
                        "preview and committed posterior updates diverged"
                    )
            receipt = PosteriorUpdateReceipt(
                intent_id=intent.intent_id,
                event_checksum=event_checksum,
                status=update_status,
                posterior_state_sha256=_posterior_fingerprint(active_posterior),
                error_kind=actual_error or None,
            )
            _worker_message(
                outbound_queue,
                work,
                kind="posterior_update",
                payload=receipt.to_dict(),
                intent_id=intent.intent_id,
            )
            _await_parent_ack(
                inbound_queue,
                expected_kind="posterior_ack",
                expected_id=intent.intent_id,
            )

            if not preview_error:
                reset_online_observations += int(active_posterior is not None)
                online_held_out_observations = next_online_held_out
                independent_observations = next_independent
            controller_frame_states = next_frame_states
            sequence_prefix = next_sequence_prefix
            events.append(sealed_event)
            if lane.split == "discovery":
                factory._remember_discovery(lane.game_id, sealed_event)
            labels = sealed_event["labels"]
            before = after
            if (
                labels["progress"]
                or labels["level_complete"]
                or labels["game_over"]
                or _science._is_terminal(after)
            ):
                stop_reason = (
                    "game_over"
                    if labels["game_over"] or _science._is_game_over(after)
                    else "progression"
                    if labels["progress"] or labels["level_complete"]
                    else "terminal"
                )
                break
        if stop_reason is None:
            stop_reason = "budget_exhausted"
    finally:
        if environment is not None:
            _operation(
                outbound_queue,
                work,
                "close",
                lambda: _science._close_runtime(runtime, environment),
            )

    final_particles, final_classes = _science._effective_posterior_capacity(
        active_posterior
    )
    return {
        "completed": completed,
        "stop_reason": stop_reason or "worker_incomplete",
        "action_count": len(events),
        "event_ids": [str(event.get("event_id", "")) for event in events],
        "online_observations": reset_online_observations,
        "error_count": reset_error_count,
        "initial_particle_count": initial_particles,
        "initial_class_count": initial_classes,
        "final_particle_count": final_particles,
        "final_class_count": final_classes,
        "continuation": {
            "independent_schema_counts": _serialize_schema_counter(
                independent_counts
            ),
            "learned_schema_counts": _serialize_schema_counter(learned_counts),
            "grounding_counts": {
                key: int(value) for key, value in sorted(grounding_counts.items())
            },
            "learned_reset_count": learned_reset_count,
            "online_held_out_observations": online_held_out_observations,
            "independent_observations": independent_observations,
        },
    }


def _selection_metadata(
    *,
    controller: str,
    posterior_metadata: Mapping[str, Any],
    independent_candidates: Any,
    independent_refusal: str,
    sequence_ranking: Any,
    decision_metadata: Mapping[str, Any],
    online_held_out_observations: int,
    independent_observations: int,
    active_posterior: Any | None,
) -> dict[str, Any]:
    independent_bank = isinstance(independent_candidates, FactorizedCandidateBank)
    return {
        "cross_fit_model": (
            "gauge_decision_engine_online_option"
            if controller == "learned" and posterior_metadata.get("posterior_used")
            else (
                "capacity_matched_factorized_independent_posterior"
                if controller == "capacity_matched_independent"
                else "none"
            )
        ),
        "posterior_candidate_count": int(
            posterior_metadata.get("posterior_candidates", 0)
        ),
        "posterior_class_capacity": int(
            independent_candidates.metrics.target_classes
            if controller == "capacity_matched_independent" and independent_bank
            else posterior_metadata.get("posterior_classes", 0)
        ),
        "posterior_observation_count": int(
            posterior_metadata.get("posterior_observations", 0)
        ),
        "online_posterior_observation_index": (
            online_held_out_observations
            if controller == "learned"
            else independent_observations
        ),
        "compatible_option_sequence_mass": (
            sequence_ranking.compatible_posterior_mass
            if sequence_ranking is not None
            else None
        ),
        "compatible_option_sequence_signatures": (
            sequence_ranking.compatible_signature_count
            if sequence_ranking is not None
            else 0
        ),
        "ranked_option_sequence_signatures": (
            sequence_ranking.signature_count if sequence_ranking is not None else 0
        ),
        "explicit_option_sequence_mass": (
            sequence_ranking.explicit_posterior_mass
            if sequence_ranking is not None
            else None
        ),
        "residual_option_sequence_mass": (
            sequence_ranking.residual_posterior_mass
            if sequence_ranking is not None
            else None
        ),
        "decision_engine_used": bool(decision_metadata["engine_used"]),
        "decision_reason": str(decision_metadata["reason"]),
        "decision_entropy": decision_metadata["normalized_entropy"],
        "option_conditioned": bool(
            controller == "learned" and posterior_metadata.get("option_conditioned")
        ),
        "posterior_family": (
            "strict_five_factor_variational_control"
            if isinstance(active_posterior, FactorizedGaugeProgramPosterior)
            else "joint_gauge_posterior"
            if active_posterior is not None
            else "none"
        ),
        "factorized_bank_capacity_matched": bool(
            independent_bank and independent_candidates.metrics.capacity_matched
        ),
        "factorized_target_particles": (
            independent_candidates.metrics.target_particles if independent_bank else 0
        ),
        "factorized_target_classes": (
            independent_candidates.metrics.target_classes if independent_bank else 0
        ),
        "factorized_mdl_prior_preserved": isinstance(
            active_posterior, FactorizedGaugeProgramPosterior
        ),
        "factorized_control_refusal": independent_refusal,
    }


def _all_reset_reports(
    journal: DurableCollectionJournal,
) -> tuple[ResetReport, ...]:
    return tuple(
        report
        for lane in source_lane_registry()
        for work in reset_work_specs(lane)
        if (report := journal.read_reset_report(work)) is not None
    )


def _completed_discovery_events(
    journal: DurableCollectionJournal,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for lane in source_lane_registry():
        if lane.split != "discovery":
            continue
        for work in reset_work_specs(lane):
            report = journal.read_reset_report(work)
            if report is not None and report.status == "COMPLETE":
                rows.extend(journal.events_for_reset(work))
    return tuple(rows)


_LOCAL_RESET_STOP_REASONS = frozenset(
    {
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "interrupted_before_reset_commit",
        "parent_interrupted",
    }
)


def _collection_stop_scope(reason: str) -> Literal["local_reset", "global"]:
    """Classify whether a sealed reset miss may leave later work runnable."""

    return "local_reset" if str(reason) in _LOCAL_RESET_STOP_REASONS else "global"


def _continuation_before(
    journal: DurableCollectionJournal,
    work: ResetWorkSpec,
) -> dict[str, Any]:
    continuation: dict[str, Any] = {}
    for prior in reset_work_specs(work.lane):
        if prior.reset_index >= work.reset_index:
            break
        report = journal.read_reset_report(prior)
        if report is None:
            raise ResumeRefusalError(
                "reset execution order lacks a predecessor checkpoint"
            )
        if report.status == "COMPLETE":
            continuation = dict(report.continuation)
        elif _collection_stop_scope(report.stop_reason) != "local_reset":
            raise ResumeRefusalError(
                "reset execution order crosses a global predecessor failure"
            )
    return continuation


def _attest_unresolved_after_worker(
    journal: DurableCollectionJournal,
    work: ResetWorkSpec,
    *,
    reason: Literal[
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "worker_failed",
        "parent_interrupted",
        "environment_call_unattestable",
    ],
) -> None:
    for intent in journal.intents_for_reset(work):
        if journal._event_path(intent).is_file():
            continue
        if journal._unresolved_path(intent).is_file():
            continue
        journal.record_unresolved_intent(intent=intent, reason=reason)


def _reset_report_from_outcome(
    *,
    journal: DurableCollectionJournal,
    work: ResetWorkSpec,
    outcome: WorkerOutcome,
    prior_continuation: Mapping[str, Any],
) -> ResetReport:
    if outcome.status == "HARD_TIMEOUT":
        _attest_unresolved_after_worker(
            journal, work, reason="hard_reset_timeout"
        )
    elif outcome.status == "FAILED":
        _attest_unresolved_after_worker(journal, work, reason="worker_failed")
    accounting = journal.reset_accounting(work)
    events = journal.events_for_reset(work)
    complete = bool(
        outcome.status == "COMPLETE"
        and outcome.payload.get("completed") is True
        and accounting.equation_holds
        and accounting.explicitly_unresolved_intent_count == 0
        and accounting.unknown_intent_count == 0
        and accounting.posterior_update_count == accounting.sealed_event_count
    )
    status: ReportStatus = (
        "COMPLETE"
        if complete
        else "UNATTESTABLE"
        if (
            accounting.explicitly_unresolved_intent_count
            or accounting.unknown_intent_count
        )
        else "ABORTED"
    )
    raw_continuation = outcome.payload.get("continuation")
    continuation = (
        dict(_mapping(raw_continuation))
        if isinstance(raw_continuation, Mapping)
        else dict(_jsonable(prior_continuation))
    )
    continuation["audit"] = {
        "reset_index": work.reset_index,
        "controller": work.controller,
        "action_count": accounting.sealed_event_count,
        "online_observations": int(outcome.payload.get("online_observations", 0)),
        "error_count": int(outcome.payload.get("error_count", 0)),
        "initial_particle_count": int(
            outcome.payload.get("initial_particle_count", 0)
        ),
        "initial_class_count": int(outcome.payload.get("initial_class_count", 0)),
        "final_particle_count": int(outcome.payload.get("final_particle_count", 0)),
        "final_class_count": int(outcome.payload.get("final_class_count", 0)),
        "stop_reason": str(outcome.payload.get("stop_reason", "worker_incomplete")),
    }
    report = ResetReport(
        work=work,
        status=status,
        issued_intents=accounting.authorized_intent_count,
        sealed_events=accounting.sealed_event_count,
        unresolved_intents=accounting.explicitly_unresolved_intent_count,
        posterior_updates=accounting.posterior_update_count,
        elapsed_seconds=outcome.elapsed_seconds,
        stop_reason=str(outcome.payload.get("stop_reason", "worker_incomplete")),
        event_ids_sha256=canonical_sha256(
            [str(event.get("event_id", "")) for event in events]
        ),
        continuation=continuation,
    )
    journal.write_reset_report(report)
    return report


def _cross_fit_unit_for_lane(
    lane: SourceLaneKey,
    resets: Sequence[ResetReport],
    discovery_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if lane.split == "discovery":
        return None
    training_games = tuple(game for game in SOURCE_GAMES if game != lane.game_id)
    donor_ids = [
        str(event.get("event_id", ""))
        for event in discovery_events
        if str(event.get("game_id", "")) in training_games
    ]
    return {
        "held_out_game": lane.game_id,
        "seed": lane.seed,
        "training_games": list(training_games),
        "donor_event_count": len(donor_ids),
        "donor_event_ids_sha256": _t10_2_protocol.canonical_sha256(donor_ids),
        "held_out_prefit_events_used": 0,
        "resets": [dict(_mapping(item.continuation.get("audit", {}))) for item in resets],
    }


def _finalize_lane_report(
    journal: DurableCollectionJournal,
    lane: SourceLaneKey,
    *,
    elapsed_seconds: float | None = None,
    timing_admissible: bool = True,
) -> LaneReport:
    resets = tuple(
        report
        for work in reset_work_specs(lane)
        if (report := journal.read_reset_report(work)) is not None
    )
    complete = bool(
        len(resets) == SOURCE_RESETS_PER_LANE
        and all(item.status == "COMPLETE" for item in resets)
        and not any(item.unresolved_intents for item in resets)
        and timing_admissible
    )
    status: ReportStatus = (
        "COMPLETE"
        if complete
        else "UNATTESTABLE"
        if any(item.status == "UNATTESTABLE" for item in resets)
        else "ABORTED"
    )
    discovery = _completed_discovery_events(journal)
    reset_elapsed_seconds = sum(item.elapsed_seconds for item in resets)
    reconstructed_elapsed_seconds = (
        reset_elapsed_seconds
        if elapsed_seconds is None
        else max(reset_elapsed_seconds, float(elapsed_seconds))
    )
    if (
        not math.isfinite(reconstructed_elapsed_seconds)
        or reconstructed_elapsed_seconds < 0.0
    ):
        raise JournalIntegrityError("lane duration is invalid")
    report = LaneReport(
        lane=lane,
        status=status,
        resets=resets,
        issued_intents=sum(item.issued_intents for item in resets),
        sealed_events=sum(item.sealed_events for item in resets),
        unresolved_intents=sum(item.unresolved_intents for item in resets),
        elapsed_seconds=reconstructed_elapsed_seconds,
        cross_fit_unit=(
            _cross_fit_unit_for_lane(lane, resets, discovery) if complete else None
        ),
    )
    journal.write_lane_report(report)
    return report


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return _t10_2_protocol.artifact_descriptor(path)


def _build_cross_fit_audit(
    *,
    manifest: Mapping[str, Any],
    source_event_path: Path,
    source_events: Sequence[Mapping[str, Any]],
    lane_reports: Sequence[LaneReport],
    factory: T10_2_1SourceFactory,
) -> dict[str, Any]:
    from . import t10_2_1_protocol as protocol

    units = _t10_2_protocol._canonical_cross_fit_units(
        [
            dict(report.cross_fit_unit)
            for report in lane_reports
            if report.cross_fit_unit is not None
        ]
    )
    factory_binding = protocol._factory_binding(factory, manifest)
    # The frozen reader recomputes this exact mapping while its T10.2 globals
    # are temporarily rebound to the T10.2.1 preregistration.  Build under the
    # same compatibility context so the alternating A/B/A/B reset schedule is
    # checked identically at collection and compile time.
    with protocol._legacy_bindings():
        checks = protocol._legacy_cross_fit_checks(
            manifest=manifest,
            source_events=source_events,
            factory=factory_binding,
            units=units,
        )
    payload = {
        "format_version": CROSS_FIT_AUDIT_FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "source_events": _artifact_descriptor(source_event_path),
        "source_event_ids_sha256": _t10_2_protocol.canonical_sha256(
            [str(event.get("event_id", "")) for event in source_events]
        ),
        "factory": factory_binding,
        "registered_unit_count": len(units),
        "units": units,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return _signed(payload, checksum_key="audit_checksum")


def _write_event_ledger(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    _t10_2_protocol.write_event_ledger(path, events)


def _read_existing_collection_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_canonical_json(path)
    _verify_signed(payload, checksum_key="report_checksum")
    return payload


_COLLECTION_ROOT_ALLOWLIST: Mapping[str, Literal["file", "directory"]] = {
    ".active-collector.lock": "file",
    INVOCATION_STATE_FILENAME: "file",
    INVOCATION_TERMINAL_FILENAME: "file",
    JOURNAL_DIRECTORY_NAME: "directory",
    CHECKPOINT_FILENAME: "file",
    SOURCE_EVENTS_FILENAME: "file",
    CROSS_FIT_AUDIT_FILENAME: "file",
    COLLECTION_REPORT_FILENAME: "file",
}


class _CollectionLease:
    """Crash-releasable OS lock; the inert lock file may persist safely."""

    def __init__(self, path: Path, handle: Any) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def acquire(cls, path: Path) -> "_CollectionLease":
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise JournalIntegrityError("collection lease path is not a regular file")
        handle = path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as exc:
            handle.close()
            raise JournalConflictError("collection lease is held elsewhere") from exc
        _fsync_directory(path.parent)
        return cls(path, handle)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def _read_invocation_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    state = _read_canonical_json(path)
    _verify_signed(state, checksum_key="state_checksum")
    if state.get("format_version") != INVOCATION_STATE_FORMAT_VERSION:
        raise JournalIntegrityError("collection invocation state format drifted")
    if state.get("status") != "OPEN":
        raise JournalIntegrityError("collection invocation state status drifted")
    _require_digest(str(state.get("manifest_checksum", "")), label="manifest checksum")
    _require_digest(str(state.get("invocation_id", "")), label="invocation id")
    cumulative = float(state.get("cumulative_active_seconds", -1.0))
    if not math.isfinite(cumulative) or cumulative < 0.0:
        raise JournalIntegrityError("collection invocation duration is invalid")
    base = float(state.get("base_cumulative_active_seconds", -1.0))
    if not math.isfinite(base) or base < 0.0 or cumulative < base:
        raise JournalIntegrityError("collection invocation base duration is invalid")
    for field_name in ("monotonic_started", "wall_monotonic_started"):
        observed = float(state.get(field_name, -1.0))
        if not math.isfinite(observed) or observed < 0.0:
            raise JournalIntegrityError(
                f"collection invocation {field_name} is invalid"
            )
    return state


def _write_invocation_state(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    state = _signed(payload, checksum_key="state_checksum")
    _write_once(path, state)
    reloaded = _read_invocation_state(path)
    if reloaded != state:
        raise JournalIntegrityError("collection invocation state failed reconstruction")
    return state


def _read_invocation_terminal(
    path: Path,
    *,
    opened: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    terminal = _read_canonical_json(path)
    _verify_signed(terminal, checksum_key="terminal_checksum")
    if terminal.get("format_version") != INVOCATION_TERMINAL_FORMAT_VERSION:
        raise JournalIntegrityError("collection invocation terminal format drifted")
    if terminal.get("status") not in {"CLOSED", "HARD_TIMEOUT"}:
        raise JournalIntegrityError("collection invocation terminal status drifted")
    _require_digest(
        str(terminal.get("manifest_checksum", "")),
        label="manifest checksum",
    )
    _require_digest(str(terminal.get("invocation_id", "")), label="invocation id")
    _require_digest(
        str(terminal.get("open_state_checksum", "")),
        label="open invocation checksum",
    )
    cumulative = float(terminal.get("cumulative_active_seconds", -1.0))
    if not math.isfinite(cumulative) or cumulative < 0.0:
        raise JournalIntegrityError("collection terminal duration is invalid")
    terminal_monotonic = float(terminal.get("terminal_monotonic", -1.0))
    if not math.isfinite(terminal_monotonic) or terminal_monotonic < 0.0:
        raise JournalIntegrityError("collection terminal monotonic time is invalid")
    report_core_checksum = terminal.get("report_core_checksum")
    if report_core_checksum is not None:
        _require_digest(
            str(report_core_checksum),
            label="projected terminal report core checksum",
        )
    if opened is not None:
        if terminal.get("manifest_checksum") != opened.get("manifest_checksum"):
            raise JournalIntegrityError("collection terminal escaped its manifest")
        if terminal.get("invocation_id") != opened.get("invocation_id"):
            raise JournalIntegrityError("collection terminal escaped its invocation")
        if terminal.get("open_state_checksum") != opened.get("state_checksum"):
            raise JournalIntegrityError("collection terminal escaped its OPEN receipt")
    return terminal


def _claim_invocation_terminal(
    path: Path,
    *,
    opened: Mapping[str, Any],
    status: Literal["CLOSED", "HARD_TIMEOUT"],
    cumulative_active_seconds: float,
    terminal_monotonic: float | None = None,
    report_core_checksum: str | None = None,
) -> dict[str, Any]:
    """Atomically linearize CLOSED versus HARD_TIMEOUT for one collection."""

    observed_terminal = (
        time.monotonic() if terminal_monotonic is None else float(terminal_monotonic)
    )
    cumulative = float(cumulative_active_seconds)
    if not math.isfinite(cumulative) or cumulative < 0.0:
        raise JournalIntegrityError("collection terminal duration is invalid")
    if report_core_checksum is not None:
        _require_digest(
            report_core_checksum,
            label="projected terminal report core checksum",
        )
    payload = _signed(
        {
            "format_version": INVOCATION_TERMINAL_FORMAT_VERSION,
            "manifest_checksum": opened["manifest_checksum"],
            "invocation_id": opened["invocation_id"],
            "status": status,
            "open_state_checksum": opened["state_checksum"],
            "cumulative_active_seconds": cumulative,
            "terminal_monotonic": observed_terminal,
            "report_core_checksum": report_core_checksum,
        },
        checksum_key="terminal_checksum",
    )
    try:
        _write_once(path, payload)
    except JournalConflictError:
        # A concurrent sentinel or collector owns the immutable linearization
        # point.  Never overwrite it; reconstruct the winner instead.
        deadline = time.monotonic() + 1.0
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
    winner = _read_invocation_terminal(path, opened=opened)
    if winner is None:
        raise JournalConflictError("collection terminal claim has no durable winner")
    return winner


def _record_global_hard_timeout(
    path: Path,
    terminal_path: Path | None = None,
) -> dict[str, Any] | None:
    try:
        opened = _read_invocation_state(path)
        if opened is None:
            return None
        return _claim_invocation_terminal(
            (
                terminal_path
                if terminal_path is not None
                else path.with_name(INVOCATION_TERMINAL_FILENAME)
            ),
            opened=opened,
            status="HARD_TIMEOUT",
            cumulative_active_seconds=COLLECTION_ABSOLUTE_SECONDS,
            terminal_monotonic=time.monotonic(),
        )
    except Exception:
        # Process-tree termination remains mandatory even if the receipt cannot
        # be persisted; the surviving OPEN state is itself fail-closed on resume.
        return None


def _terminal_absolute_wall_bound(
    opened: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> bool:
    cumulative = float(terminal["cumulative_active_seconds"])
    if not math.isfinite(cumulative) or cumulative < 0.0:
        return False
    if terminal.get("status") == "CLOSED":
        return cumulative <= COLLECTION_ABSOLUTE_SECONDS
    # A CAS-winning HARD_TIMEOUT is the durable proof that the independent
    # sentinel enforced the registered absolute bound.  Unlike the old clamp,
    # this cannot be manufactured by report arithmetic after the deadline.
    return bool(
        terminal.get("status") == "HARD_TIMEOUT"
        and cumulative >= COLLECTION_ABSOLUTE_SECONDS
    )


def _invocation_report_binding(
    opened: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": INVOCATION_TERMINAL_FORMAT_VERSION,
        "invocation_id": opened["invocation_id"],
        "open_state_checksum": opened["state_checksum"],
        "terminal_checksum": terminal["terminal_checksum"],
        "terminal_status": terminal["status"],
        "report_core_checksum": terminal.get("report_core_checksum"),
        "absolute_wall_bound": _terminal_absolute_wall_bound(opened, terminal),
    }


def _validate_report_invocation_binding(
    report: Mapping[str, Any],
    *,
    opened: Mapping[str, Any] | None,
    terminal: Mapping[str, Any] | None,
    required: bool,
) -> None:
    raw_binding = report.get("invocation")
    if raw_binding is None and not required:
        return
    if opened is None or terminal is None:
        raise JournalIntegrityError("terminal report lacks durable invocation receipts")
    binding = _mapping(raw_binding)
    expected = _invocation_report_binding(opened, terminal)
    if dict(binding) != expected:
        raise JournalIntegrityError("terminal report invocation binding drifted")
    projected_core = terminal.get("report_core_checksum")
    if required and terminal.get("status") == "CLOSED":
        _require_digest(
            str(projected_core or ""),
            label="projected terminal report core checksum",
        )
        report_core = dict(report)
        report_core.pop("report_checksum", None)
        report_core.pop("invocation", None)
        if canonical_sha256(report_core) != projected_core:
            raise JournalIntegrityError(
                "terminal report escaped its CLOSED projected checksum"
            )
    if (
        report.get("status") == "T10_2_1_SOURCE_COLLECTION_COMPLETE"
        and terminal.get("status") != "CLOSED"
    ):
        raise JournalIntegrityError("successful report followed a hard timeout")


def _collection_hard_watchdog_entry(
    parent_pid: int,
    deadline_monotonic: float,
    cancel_event: Any,
    invocation_state_path: str,
    invocation_terminal_path: str | None = None,
) -> None:
    """External wall-clock sentinel; Windows termination covers descendants."""

    while True:
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0.0:
            break
        if cancel_event.wait(timeout=min(1.0, remaining)):
            return
        if os.name != "nt":
            try:
                os.kill(int(parent_pid), 0)
            except OSError:
                return
    if cancel_event.is_set():
        return
    terminal = _record_global_hard_timeout(
        Path(invocation_state_path),
        (
            None
            if invocation_terminal_path is None
            else Path(invocation_terminal_path)
        ),
    )
    # CLOSED wins the CAS only after a projected report core exists.  At the
    # absolute deadline the matching report must already be durable: there is
    # no post-deadline grace, because 5,400 s is a strict collection wall.
    if terminal is not None and terminal.get("status") == "CLOSED":
        state_path = Path(invocation_state_path)
        report_path = state_path.with_name(COLLECTION_REPORT_FILENAME)
        try:
            opened = _read_invocation_state(state_path)
            report = _read_existing_collection_report(report_path)
            if opened is not None and report is not None:
                is_minimal_data = bool(
                    report.get("status") == "DATA_OR_PROVENANCE_INVALID"
                    and _mapping(report.get("events", {})).get("available")
                    is False
                )
                _validate_report_invocation_binding(
                    report,
                    opened=opened,
                    terminal=terminal,
                    required=not is_minimal_data,
                )
                return
        except Exception:
            pass
    if os.name == "nt":
        try:
            completed = subprocess.run(
                [
                    "taskkill.exe",
                    "/PID",
                    str(int(parent_pid)),
                    "/T",
                    "/F",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=10.0,
            )
            if getattr(completed, "returncode", 1) == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            import signal

            os.kill(int(parent_pid), signal.SIGTERM)
        except OSError:
            pass
        return
    # T10.2.1 production collection is registered for Windows spawn.  The
    # non-Windows fallback terminates the blocked collector itself; process-tree
    # termination is only part of the registered Windows production contract.
    try:
        import signal

        os.kill(int(parent_pid), signal.SIGKILL)
    except OSError:
        return


def _reset_hard_watchdog_entry(
    collector_pid: int,
    deadline_monotonic: float,
    cancel_event: Any,
) -> None:
    while True:
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0.0:
            break
        if cancel_event.wait(timeout=min(0.25, remaining)):
            return
    if cancel_event.is_set():
        return
    if os.name == "nt":
        try:
            completed = subprocess.run(
                [
                    "taskkill.exe",
                    "/PID",
                    str(int(collector_pid)),
                    "/T",
                    "/F",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=10.0,
            )
            if getattr(completed, "returncode", 1) == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            import signal

            os.kill(int(collector_pid), signal.SIGTERM)
        except OSError:
            return


def _assert_collection_root_allowlist(
    destination: Path,
    *,
    active_lease: Path | None = None,
) -> None:
    """Refuse unregistered collection-root topology before physical action."""

    unexpected: list[str] = []
    for child in destination.iterdir():
        if (
            active_lease is not None
            and child == active_lease
            and child.is_file()
            and not child.is_symlink()
        ):
            continue
        expected_kind = _COLLECTION_ROOT_ALLOWLIST.get(child.name)
        if expected_kind is None or child.is_symlink():
            unexpected.append(child.name)
            continue
        if expected_kind == "file" and not child.is_file():
            unexpected.append(child.name)
        elif expected_kind == "directory" and not child.is_dir():
            unexpected.append(child.name)
    if unexpected:
        raise JournalIntegrityError(
            "unregistered collection-root artifacts: "
            + ",".join(sorted(unexpected))
        )


def _publish_minimal_data_report(
    *,
    destination: Path,
    manifest_checksum: str,
    started: float,
    clock: Callable[[], float],
    error: BaseException,
    limits: Any,
    invocation_open: Mapping[str, Any] | None = None,
    invocation_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a parseable DATA terminal when durable reconstruction is impossible."""

    report_path = destination / COLLECTION_REPORT_FILENAME
    existing = _read_existing_collection_report(report_path)
    if existing is not None:
        return existing
    finished = float(clock())
    if not math.isfinite(finished) or finished < started:
        finished = started
    accounting = {
        "authorized_intent_count": 0,
        "sealed_event_count": 0,
        "explicitly_unresolved_intent_count": 0,
        "unknown_intent_count": 1,
        "maximum_authorized_intents": SOURCE_MAXIMUM_AUTHORIZED_INTENTS,
        "equation_holds": False,
    }
    checks = {
        "action_equation_holds": False,
        "no_unknown_intents": False,
        "no_unresolved_intents": True,
        "authorized_action_cap": True,
        "sealed_events_bound": False,
        "all_registered_lanes_complete": False,
        "all_registered_resets_complete": False,
        "journal_reconstructed": False,
        "checkpoint_reconstructed": False,
        "physical_steps_not_replayed": True,
        # A minimal DATA report cannot reconstruct the durable timing proof.
        # Do not turn a short current call into a claim about the global bound.
        "absolute_wall_bound": False,
        "cross_fit_schedule": False,
        "source_firewall_closed": True,
    }
    payload = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "phase": "collect",
        "status": "DATA_OR_PROVENANCE_INVALID",
        "manifest_checksum": manifest_checksum,
        "games": list(SOURCE_GAMES),
        "splits": {
            "discovery": list(DISCOVERY_SEEDS),
            "leave_one_game_out_confirmation": list(CONFIRMATION_SEEDS),
        },
        "event_count": 0,
        "authorized_action_count": 0,
        "unresolved_intent_count": 0,
        "action_accounting": accounting,
        "durability": {
            "lane_report_count": 0,
            "reset_report_count": 0,
            "journal_reconstructed": False,
            "checkpoint_reconstructed": False,
            "physical_steps_replayed_on_resume": 0,
        },
        "checks": checks,
        "events": {
            "available": False,
            "reason": "unreconstructible",
        },
        "cross_fit_audit": {
            "available": False,
            "reason": "unreconstructible",
        },
        "cross_fit_checks": {},
        "timing": {
            "clock": "time.perf_counter",
            "current_invocation_started": started,
            "current_invocation_finished": finished,
            "current_invocation_seconds": finished - started,
            "cumulative_active_seconds": finished - started,
            "stop_new_actions_seconds": COLLECTION_COOPERATIVE_SECONDS,
            "absolute_seconds": COLLECTION_ABSOLUTE_SECONDS,
        },
        "resource_preflight": None,
        "terminal_reason": "data_or_provenance_error",
        "error": {
            "kind": type(error).__name__,
            "message_sha256": canonical_sha256(str(error)),
        },
        "factory": None,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    if invocation_open is not None and invocation_terminal is not None:
        payload["invocation"] = _invocation_report_binding(
            invocation_open,
            invocation_terminal,
        )
    report = _signed(payload, checksum_key="report_checksum")
    encoded_size = len((canonical_json(report) + "\n").encode("utf-8"))
    if encoded_size > limits.maximum_derived_file_bytes:
        raise _t10_2_protocol.ResourceGateError(
            "minimal collection DATA report exceeds registered size"
        )
    _write_once(report_path, report)
    _t10_2_protocol.enforce_artifact_limit(
        report_path,
        kind="derived",
        limits=limits,
    )
    return report


def _collect_phase_impl(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None,
    env_factory: T10_2_1SourceFactory | None,
    resource_probe: Callable[[str | Path | None], Any] = (
        _t10_2_protocol.resource_snapshot
    ),
    limits: Any = _t10_2_protocol.DEFAULT_RESOURCE_LIMITS,
    clock: Callable[[], float] = time.perf_counter,
    process_context: Any = None,
    _started_override: float | None = None,
) -> dict[str, Any]:
    """Collect the 18 registered source lanes with durable reset commits.

    Rerunning this function never physically replays a partial reset.  Complete
    reset/lane reports are verified and skipped; a reset containing intents but
    lacking its commit marker is terminally attested and the acquisition closes.
    """

    from . import t10_2_1_protocol as protocol

    started = float(
        clock() if _started_override is None else _started_override
    )
    if not math.isfinite(started):
        raise WorkerProtocolError("collection monotonic start is invalid")
    root = Path(repo_root or protocol._repo_root()).resolve()
    # A terminal artifact is never written into the registered namespace until
    # the manifest itself has authenticated and supplied its signed checksum.
    manifest = protocol.load_manifest(manifest_path, repo_root=root)
    destination = protocol._registered_output_dir(
        manifest=manifest,
        output_dir=output_dir,
        repo_root=root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / COLLECTION_REPORT_FILENAME
    existing_report = _read_existing_collection_report(report_path)
    if existing_report is not None:
        if existing_report.get("manifest_checksum") != manifest["manifest_checksum"]:
            raise JournalIntegrityError("existing collection report escaped manifest")
        is_minimal_data = bool(
            existing_report.get("status") == "DATA_OR_PROVENANCE_INVALID"
            and _mapping(existing_report.get("events", {})).get("available")
            is False
            and _mapping(existing_report.get("cross_fit_audit", {})).get(
                "available"
            )
            is False
        )
        if is_minimal_data and existing_report.get("invocation") is None:
            _t10_2_protocol.enforce_output_artifacts(destination, limits=limits)
            return existing_report
        existing_open = _read_invocation_state(
            destination / INVOCATION_STATE_FILENAME
        )
        existing_terminal = _read_invocation_terminal(
            destination / INVOCATION_TERMINAL_FILENAME,
            opened=existing_open,
        )
        _validate_report_invocation_binding(
            existing_report,
            opened=existing_open,
            terminal=existing_terminal,
            required=not is_minimal_data,
        )
        if is_minimal_data:
            # DATA is itself the durable terminal proof that reconstruction was
            # impossible.  Never touch the journal/checkpoint on this fast path:
            # their corruption is the evidence boundary, not a reason to make
            # repeated invocations non-idempotent.
            _t10_2_protocol.enforce_output_artifacts(destination, limits=limits)
            return existing_report
        journal = DurableCollectionJournal(
            destination / JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(manifest["manifest_checksum"]),
        )
        initial_checkpoint = journal.reconstruct_checkpoint()
        if not protocol._collection_durable_evidence_matches(
            existing_report,
            manifest=manifest,
            output_dir=destination,
        ):
            raise JournalIntegrityError(
                "terminal collection report did not reconstruct from its journal"
            )
        if (
            _mapping(existing_report.get("durability", {})).get(
                "checkpoint_checksum"
            )
            != initial_checkpoint.checkpoint_checksum
        ):
            raise JournalIntegrityError(
                "terminal collection checkpoint checksum drifted"
            )
        _t10_2_protocol.enforce_output_artifacts(destination, limits=limits)
        return existing_report

    lease_path = destination / ".active-collector.lock"
    lease = _CollectionLease.acquire(lease_path)
    invocation_state_path = destination / INVOCATION_STATE_FILENAME
    invocation_terminal_path = destination / INVOCATION_TERMINAL_FILENAME
    invocation_state: dict[str, Any] | None = None
    invocation_terminal: dict[str, Any] | None = None
    resume_budget_exhausted = False
    recovered_active_seconds = 0.0
    try:
        invocation_state = _read_invocation_state(invocation_state_path)
        if invocation_state is None:
            wall_started = time.monotonic()
            invocation_id = canonical_sha256(
                {
                    "manifest_checksum": manifest["manifest_checksum"],
                    "monotonic_started": started,
                    "wall_monotonic_started": wall_started,
                    "process_id": os.getpid(),
                }
            )
            invocation_state = _write_invocation_state(
                invocation_state_path,
                {
                    "format_version": INVOCATION_STATE_FORMAT_VERSION,
                    "manifest_checksum": manifest["manifest_checksum"],
                    "invocation_id": invocation_id,
                    "status": "OPEN",
                    "base_cumulative_active_seconds": 0.0,
                    "cumulative_active_seconds": 0.0,
                    "monotonic_started": started,
                    "wall_monotonic_started": wall_started,
                },
            )
        if invocation_state.get("manifest_checksum") != manifest["manifest_checksum"]:
            raise JournalIntegrityError("collection invocation escaped its manifest")
        invocation_terminal = _read_invocation_terminal(
            invocation_terminal_path,
            opened=invocation_state,
        )
        recovered_active_seconds = float(
            invocation_state["cumulative_active_seconds"]
        )
        if invocation_terminal is not None:
            if invocation_terminal.get("status") == "CLOSED":
                raise JournalIntegrityError(
                    "CLOSED invocation has no immutable terminal report"
                )
            resume_budget_exhausted = True
            recovered_active_seconds = max(
                recovered_active_seconds,
                float(invocation_terminal["cumulative_active_seconds"]),
            )
        journal = DurableCollectionJournal(
            destination / JOURNAL_DIRECTORY_NAME,
            manifest_checksum=str(manifest["manifest_checksum"]),
        )
        initial_checkpoint = journal.reconstruct_checkpoint()
        recovered_active_seconds = max(
            recovered_active_seconds,
            initial_checkpoint.cumulative_active_seconds,
        )
        if (
            not resume_budget_exhausted
            and initial_checkpoint.open_lane_id is not None
        ):
            interrupted_lane = next(
                lane
                for lane in source_lane_registry()
                if lane.lane_id == initial_checkpoint.open_lane_id
            )
            if journal.read_lane_report(interrupted_lane) is not None:
                initial_checkpoint = journal.reconstruct_checkpoint(
                    cumulative_active_seconds=recovered_active_seconds,
                    close_open_lane=True,
                )
            else:
                # Charge only durable active work plus one conservative
                # reset-sized reservation.  Time while Codex/the IDE was
                # stopped is downtime, not collection time.  Advancing lane
                # time makes the reservation durable and bounds repeated
                # interruptions by the 250 s lane gate.
                interruption_reservation = min(
                    RESET_HARD_SECONDS,
                    max(
                        0.0,
                        LANE_HARD_SECONDS
                        - initial_checkpoint.open_lane_elapsed_seconds,
                    ),
                )
                if interruption_reservation > 0.0:
                    recovered_active_seconds += interruption_reservation
                    initial_checkpoint = journal.reconstruct_checkpoint(
                        cumulative_active_seconds=recovered_active_seconds,
                        open_lane=interrupted_lane,
                        open_lane_elapsed_seconds=(
                            initial_checkpoint.open_lane_elapsed_seconds
                            + interruption_reservation
                        ),
                    )
        else:
            initial_checkpoint = journal.reconstruct_checkpoint(
                cumulative_active_seconds=recovered_active_seconds,
            )
        if (
            invocation_terminal is None
            and recovered_active_seconds >= COLLECTION_ABSOLUTE_SECONDS
        ):
            invocation_terminal = _claim_invocation_terminal(
                invocation_terminal_path,
                opened=invocation_state,
                status="HARD_TIMEOUT",
                cumulative_active_seconds=recovered_active_seconds,
                terminal_monotonic=time.monotonic(),
            )
            resume_budget_exhausted = True
        _assert_collection_root_allowlist(destination)
        initial_accounting = journal.accounting()
        if (
            initial_accounting.unknown_intent_count
            or not initial_accounting.equation_holds
        ):
            raise JournalIntegrityError(
                "journal topology or action equation is invalid before collection"
            )
    except Exception as exc:
        if report_path.exists():
            lease.release()
            raise
        try:
            if invocation_state is not None and invocation_terminal is None:
                terminal_now = time.monotonic()
                invocation_terminal = _claim_invocation_terminal(
                    invocation_terminal_path,
                    opened=invocation_state,
                    status=(
                        "CLOSED"
                        if recovered_active_seconds
                        <= COLLECTION_ABSOLUTE_SECONDS
                        else "HARD_TIMEOUT"
                    ),
                    cumulative_active_seconds=recovered_active_seconds,
                    terminal_monotonic=terminal_now,
                )
            return _publish_minimal_data_report(
                destination=destination,
                manifest_checksum=str(manifest["manifest_checksum"]),
                started=started,
                clock=clock,
                error=exc,
                limits=limits,
                invocation_open=invocation_state,
                invocation_terminal=invocation_terminal,
            )
        finally:
            lease.release()

    prior_active_seconds = recovered_active_seconds

    def observed_active_seconds() -> float:
        now = float(clock())
        if not math.isfinite(now) or now < started:
            raise WorkerProtocolError("collection monotonic clock regressed")
        observed = prior_active_seconds + (now - started)
        return max(prior_active_seconds, observed)

    def claim_terminal_receipt(
        cumulative_active_seconds: float,
        *,
        report_core_checksum: str | None = None,
    ) -> dict[str, Any]:
        nonlocal invocation_terminal
        if invocation_state is None:
            raise JournalIntegrityError("collection invocation OPEN receipt is absent")
        if invocation_terminal is None:
            terminal_now = time.monotonic()
            within_wall_bound = (
                float(cumulative_active_seconds)
                <= COLLECTION_ABSOLUTE_SECONDS
            )
            invocation_terminal = _claim_invocation_terminal(
                invocation_terminal_path,
                opened=invocation_state,
                status="CLOSED" if within_wall_bound else "HARD_TIMEOUT",
                cumulative_active_seconds=cumulative_active_seconds,
                terminal_monotonic=terminal_now,
                report_core_checksum=report_core_checksum,
            )
        return invocation_terminal

    def publish_terminal_data(error: BaseException) -> dict[str, Any]:
        try:
            terminal = claim_terminal_receipt(observed_active_seconds())
            return _publish_minimal_data_report(
                destination=destination,
                manifest_checksum=str(manifest["manifest_checksum"]),
                started=started,
                clock=clock,
                error=error,
                limits=limits,
                invocation_open=invocation_state,
                invocation_terminal=terminal,
            )
        finally:
            lease.release()

    process_context = process_context or multiprocessing.get_context("spawn")
    resumed_complete_resets = 0
    resumed_complete_lanes = 0
    local_stop_reasons: list[str] = []
    terminal_reason = (
        "registered_collection_deadline" if resume_budget_exhausted else ""
    )
    data_error: BaseException | None = None
    snapshot_payload: Mapping[str, Any] | None = None
    factory: T10_2_1SourceFactory | None = None
    binding: Mapping[str, Any] | None = None
    current_lane: SourceLaneKey | None = None
    current_work: ResetWorkSpec | None = None
    lane_started = started
    lane_resumed_seconds = 0.0
    resumed_open_lane_id = initial_checkpoint.open_lane_id
    resumed_open_lane_seconds = initial_checkpoint.open_lane_elapsed_seconds
    try:
        _assert_collection_root_allowlist(
            destination,
            active_lease=lease_path,
        )
        if journal.accounting().unknown_intent_count:
            raise JournalIntegrityError("journal topology changed before collection")

        for lane in source_lane_registry():
            protocol.enforce_environment_firewall(
                phase="collect", game_id=lane.game_id
            )
        factory = env_factory or T10_2_1SourceFactory(manifest=manifest)
        if not isinstance(factory, T10_2_1SourceFactory):
            raise WorkerProtocolError(
                "production collection requires T10_2_1SourceFactory"
            )
        binding = protocol._factory_binding(factory, manifest)
        if binding.get("code_bound") is not True:
            raise WorkerProtocolError("T10.2.1 source factory is not manifest-bound")

        if not terminal_reason:
            snapshot = resource_probe(root)
            snapshot_payload = snapshot.to_dict()
            _t10_2_protocol.enforce_resource_limits(
                snapshot,
                limits=limits,
                expensive=True,
            )

        for lane in (() if terminal_reason else source_lane_registry()):
            current_lane = lane
            existing_lane = journal.read_lane_report(lane)
            if existing_lane is not None:
                if existing_lane.status == "COMPLETE":
                    resumed_complete_lanes += 1
                    resumed_complete_resets += len(existing_lane.resets)
                    continue
                incomplete = [
                    item for item in existing_lane.resets if item.status != "COMPLETE"
                ]
                prior_reason = (
                    incomplete[-1].stop_reason
                    if incomplete
                    else "cooperative_reset_deadline"
                )
                if _collection_stop_scope(prior_reason) == "local_reset":
                    local_stop_reasons.append(prior_reason)
                    resumed_complete_resets += sum(
                        item.status == "COMPLETE" for item in existing_lane.resets
                    )
                    continue
                terminal_reason = prior_reason
                break

            lane_started = float(clock())
            if not math.isfinite(lane_started):
                raise WorkerProtocolError("lane monotonic start is invalid")
            lane_resumed_seconds = (
                resumed_open_lane_seconds
                if lane.lane_id == resumed_open_lane_id
                else 0.0
            )
            journal.reconstruct_checkpoint(
                cumulative_active_seconds=observed_active_seconds(),
                open_lane=lane,
                open_lane_elapsed_seconds=lane_resumed_seconds,
            )
            lane_stopped = False
            for work in reset_work_specs(lane):
                current_work = work
                existing_reset = journal.read_reset_report(work)
                if existing_reset is not None:
                    if existing_reset.status == "COMPLETE":
                        resumed_complete_resets += 1
                        continue
                    if (
                        _collection_stop_scope(existing_reset.stop_reason)
                        == "local_reset"
                    ):
                        local_stop_reasons.append(existing_reset.stop_reason)
                        continue
                    terminal_reason = existing_reset.stop_reason
                    lane_stopped = True
                    break
                prior_continuation = _continuation_before(journal, work)
                partial = journal.reset_accounting(work)
                if partial.unknown_intent_count or not partial.equation_holds:
                    raise JournalIntegrityError(
                        "reset topology/action equation is invalid before action"
                    )
                if partial.authorized_intent_count:
                    _attest_unresolved_after_worker(
                        journal,
                        work,
                        reason="parent_interrupted",
                    )
                    outcome = WorkerOutcome(
                        status="FAILED",
                        elapsed_seconds=0.0,
                        payload={
                            "completed": False,
                            "stop_reason": "interrupted_before_reset_commit",
                            "continuation": prior_continuation,
                        },
                        error_kind="ParentInterrupted",
                    )
                else:
                    _assert_collection_root_allowlist(
                        destination,
                        active_lease=lease_path,
                    )
                    if report_path.exists():
                        raise JournalConflictError(
                            "collection report appeared before a new physical action"
                        )
                    global_accounting = journal.accounting()
                    if (
                        global_accounting.unknown_intent_count
                        or not global_accounting.equation_holds
                    ):
                        raise JournalIntegrityError(
                            "global journal became invalid before physical action"
                        )
                    lane_seconds = lane_resumed_seconds + max(
                        0.0,
                        float(clock()) - lane_started,
                    )
                    active_seconds = observed_active_seconds()
                    journal.reconstruct_checkpoint(
                        cumulative_active_seconds=active_seconds,
                        open_lane=lane,
                        open_lane_elapsed_seconds=lane_seconds,
                    )
                    discovery_events = _completed_discovery_events(journal)
                    active_seconds = observed_active_seconds()
                    lane_seconds = lane_resumed_seconds + max(
                        0.0,
                        float(clock()) - lane_started,
                    )
                    lane_remaining = (
                        LANE_HARD_SECONDS
                        - LANE_FINALIZATION_SECONDS
                        - lane_seconds
                    )
                    donor_games = {
                        str(item.get("game_id", "")) for item in discovery_events
                    }
                    required_donors = {
                        game for game in SOURCE_GAMES if game != lane.game_id
                    }
                    if (
                        lane.split == "leave_one_game_out_confirmation"
                        and not required_donors.issubset(donor_games)
                    ):
                        outcome = WorkerOutcome(
                            status="COOPERATIVE_STOP",
                            elapsed_seconds=0.0,
                            payload={
                                "completed": False,
                                "stop_reason": "cooperative_reset_deadline",
                                "continuation": prior_continuation,
                            },
                            error_kind="MissingCompletedDiscoveryDonors",
                        )
                    else:
                        outcome = factory.run_reset(
                            work=work,
                            journal=journal,
                            discovery_events=discovery_events,
                            continuation=prior_continuation,
                            process_context=process_context,
                            lane_remaining_seconds=lane_remaining,
                            cooperative_collection_remaining_seconds=(
                                COLLECTION_COOPERATIVE_SECONDS - active_seconds
                            ),
                            absolute_collection_remaining_seconds=(
                                COLLECTION_ABSOLUTE_SECONDS - active_seconds
                            ),
                            clock=clock,
                        )
                reset_report = _reset_report_from_outcome(
                    journal=journal,
                    work=work,
                    outcome=outcome,
                    prior_continuation=prior_continuation,
                )
                lane_seconds = lane_resumed_seconds + max(
                    0.0,
                    float(clock()) - lane_started,
                )
                journal.reconstruct_checkpoint(
                    cumulative_active_seconds=observed_active_seconds(),
                    open_lane=lane,
                    open_lane_elapsed_seconds=lane_seconds,
                )
                snapshot = resource_probe(root)
                snapshot_payload = snapshot.to_dict()
                _t10_2_protocol.enforce_resource_limits(
                    snapshot,
                    limits=limits,
                    expensive=True,
                )
                journal.reconstruct_checkpoint(
                    cumulative_active_seconds=observed_active_seconds(),
                    open_lane=lane,
                    open_lane_elapsed_seconds=(
                        lane_resumed_seconds
                        + max(0.0, float(clock()) - lane_started)
                    ),
                )
                if reset_report.status != "COMPLETE":
                    if (
                        _collection_stop_scope(reset_report.stop_reason)
                        == "local_reset"
                    ):
                        local_stop_reasons.append(reset_report.stop_reason)
                        continue
                    terminal_reason = reset_report.stop_reason
                    lane_stopped = True
                    break
            lane_seconds = lane_resumed_seconds + max(
                0.0,
                float(clock()) - lane_started,
            )
            timing_admissible = (
                lane_seconds + LANE_FINALIZATION_SECONDS <= LANE_HARD_SECONDS
            )
            lane_report = _finalize_lane_report(
                journal,
                lane,
                elapsed_seconds=min(
                    LANE_HARD_SECONDS,
                    lane_seconds + LANE_FINALIZATION_SECONDS,
                ),
                timing_admissible=timing_admissible,
            )
            journal.reconstruct_checkpoint(
                cumulative_active_seconds=observed_active_seconds(),
                close_open_lane=True,
            )
            current_work = None
            if not timing_admissible:
                local_stop_reasons.append("cooperative_reset_deadline")
            if lane_stopped:
                break
    except WorkerTimeoutError as exc:
        # A normal reset timeout is returned as a sealed WorkerOutcome and is
        # local.  Reaching this handler means termination itself was not
        # verified, so continuing physical work would be unsafe provenance.
        data_error = exc
        terminal_reason = "worker_exception"
    except _t10_2_protocol.ResourceGateError:
        terminal_reason = "resource_gate"
    except Exception as exc:
        data_error = exc
        terminal_reason = (
            "worker_exception"
            if isinstance(exc, WorkerProtocolError)
            else "environment_call_unattestable"
            if isinstance(exc, protocol.DataGateError)
            else "data_or_provenance_error"
        )
        if current_work is not None:
            try:
                if journal.read_reset_report(current_work) is None:
                    partial = journal.reset_accounting(current_work)
                    if partial.authorized_intent_count:
                        _attest_unresolved_after_worker(
                            journal,
                            current_work,
                            reason="worker_failed",
                        )
                        _reset_report_from_outcome(
                            journal=journal,
                            work=current_work,
                            outcome=WorkerOutcome(
                                status="FAILED",
                                elapsed_seconds=0.0,
                                payload={
                                    "completed": False,
                                    "stop_reason": terminal_reason,
                                    "continuation": _continuation_before(
                                        journal, current_work
                                    ),
                                },
                                error_kind=type(exc).__name__,
                            ),
                            prior_continuation=_continuation_before(
                                journal, current_work
                            ),
                        )
            except Exception:
                pass
    finally:
        # Keep the collection-wide OS lease through ledger/audit/checkpoint
        # finalization and the immutable terminal-report commit below.
        pass

    if not terminal_reason and local_stop_reasons:
        terminal_reason = local_stop_reasons[-1]

    try:
        if current_lane is not None and journal.read_lane_report(current_lane) is None:
            lane_seconds = lane_resumed_seconds + max(
                0.0,
                float(clock()) - lane_started,
            )
            _finalize_lane_report(
                journal,
                current_lane,
                elapsed_seconds=min(
                    LANE_HARD_SECONDS,
                    lane_seconds + LANE_FINALIZATION_SECONDS,
                ),
                timing_admissible=(
                    lane_seconds + LANE_FINALIZATION_SECONDS <= LANE_HARD_SECONDS
                    and data_error is None
                ),
            )
        checkpoint = journal.reconstruct_checkpoint(
            cumulative_active_seconds=observed_active_seconds(),
            close_open_lane=True,
        )
        accounting = journal.accounting()
        events = journal.all_events(complete_resets_only=False)
        ledger_path = destination / SOURCE_EVENTS_FILENAME
        _write_event_ledger(ledger_path, events)
        _t10_2_protocol.enforce_artifact_limit(
            ledger_path,
            kind="ledger",
            limits=limits,
        )
        lane_reports = journal.lane_reports()
        if factory is None:
            factory = env_factory or T10_2_1SourceFactory(manifest=manifest)
        if binding is None:
            binding = protocol._factory_binding(factory, manifest)
        cross_fit_audit = _build_cross_fit_audit(
            manifest=manifest,
            source_event_path=ledger_path,
            source_events=events,
            lane_reports=lane_reports,
            factory=factory,
        )
        cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
        _atomic_write_json(cross_fit_path, cross_fit_audit)
        _t10_2_protocol.enforce_artifact_limit(
            cross_fit_path,
            kind="derived",
            limits=limits,
        )
        checkpoint = journal.reconstruct_checkpoint(
            cumulative_active_seconds=observed_active_seconds(),
            close_open_lane=True,
        )
        _t10_2_protocol.enforce_output_artifacts(destination, limits=limits)
        reset_reports = _all_reset_reports(journal)
        complete_lanes = sum(item.status == "COMPLETE" for item in lane_reports)
        complete_resets = sum(item.status == "COMPLETE" for item in reset_reports)
        elapsed = checkpoint.cumulative_active_seconds
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise JournalIntegrityError("reconstructed collection duration is invalid")
    except _t10_2_protocol.ResourceGateError as exc:
        return publish_terminal_data(exc)
    except Exception as exc:
        return publish_terminal_data(exc)
    try:
        checkpoint = journal.reconstruct_checkpoint(
            cumulative_active_seconds=observed_active_seconds(),
            close_open_lane=True,
        )
        _t10_2_protocol.enforce_artifact_limit(
            journal.checkpoint_path,
            kind="checkpoint",
            limits=limits,
        )
    except Exception as exc:
        return publish_terminal_data(exc)
    finished = float(clock())
    if not math.isfinite(finished) or finished < started:
        return publish_terminal_data(
            WorkerProtocolError("collection monotonic finish is invalid")
        )
    try:
        checkpoint = journal.reconstruct_checkpoint(
            cumulative_active_seconds=max(
                prior_active_seconds + (finished - started),
                (
                    float(invocation_terminal["cumulative_active_seconds"])
                    if invocation_terminal is not None
                    else 0.0
                ),
            ),
            close_open_lane=True,
        )
        _t10_2_protocol.enforce_artifact_limit(
            journal.checkpoint_path,
            kind="checkpoint",
            limits=limits,
        )
        _t10_2_protocol.enforce_artifact_limit(
            invocation_state_path,
            kind="derived",
            limits=limits,
        )
    except Exception as exc:
        return publish_terminal_data(exc)
    elapsed = checkpoint.cumulative_active_seconds
    candidate_terminal_status = (
        str(invocation_terminal["status"])
        if invocation_terminal is not None
        else "CLOSED"
        if elapsed <= COLLECTION_ABSOLUTE_SECONDS
        else "HARD_TIMEOUT"
    )
    if candidate_terminal_status == "HARD_TIMEOUT":
        terminal_reason = "registered_collection_deadline"
    durability = {
        "journal_format_version": JOURNAL_FORMAT_VERSION,
        "lane_registry_sha256": canonical_sha256(
            [lane.to_dict() for lane in source_lane_registry()]
        ),
        "lane_report_count": len(lane_reports),
        "reset_report_count": len(reset_reports),
        "complete_lane_count": complete_lanes,
        "complete_reset_count": complete_resets,
        "resumed_complete_lane_count": resumed_complete_lanes,
        "resumed_complete_reset_count": resumed_complete_resets,
        "journal_reconstructed": checkpoint.journal_reconstructed,
        "checkpoint_reconstructed": checkpoint.checkpoint_reconstructed,
        "physical_steps_replayed_on_resume": (
            checkpoint.physical_steps_replayed_on_resume
        ),
        "journal_metadata": _artifact_descriptor(journal.root / "journal.json"),
        "checkpoint": _artifact_descriptor(journal.checkpoint_path),
        "checkpoint_checksum": checkpoint.checkpoint_checksum,
    }
    action_accounting = accounting.to_dict()
    checks = {
        "action_equation_holds": accounting.equation_holds,
        "no_unknown_intents": accounting.unknown_intent_count == 0,
        "no_unresolved_intents": (
            accounting.explicitly_unresolved_intent_count == 0
        ),
        "authorized_action_cap": (
            accounting.authorized_intent_count
            <= SOURCE_MAXIMUM_AUTHORIZED_INTENTS
        ),
        "sealed_events_bound": accounting.sealed_event_count == len(events),
        "all_registered_lanes_complete": complete_lanes == len(source_lane_registry()),
        "all_registered_resets_complete": complete_resets
        == len(source_lane_registry()) * SOURCE_RESETS_PER_LANE,
        "journal_reconstructed": checkpoint.journal_reconstructed,
        "checkpoint_reconstructed": checkpoint.checkpoint_reconstructed,
        "physical_steps_not_replayed": (
            checkpoint.physical_steps_replayed_on_resume == 0
        ),
        "absolute_wall_bound": (
            elapsed <= COLLECTION_ABSOLUTE_SECONDS
            if candidate_terminal_status == "CLOSED"
            else elapsed >= COLLECTION_ABSOLUTE_SECONDS
        ),
        "cross_fit_schedule": cross_fit_audit["passed"] is True,
        "source_firewall_closed": True,
    }
    complete_structure = bool(
        complete_lanes == len(source_lane_registry())
        and complete_resets
        == len(source_lane_registry()) * SOURCE_RESETS_PER_LANE
    )
    success = bool(
        all(checks.values())
        and all(cross_fit_audit["checks"].values())
        and data_error is None
        and candidate_terminal_status == "CLOSED"
    )
    resource_stop_reasons = {
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "registered_collection_deadline",
        "resource_gate",
        "interrupted_before_reset_commit",
        "parent_interrupted",
    }
    provenance_invalid = bool(
        data_error is not None
        or accounting.unknown_intent_count
        or not accounting.equation_holds
        or binding is None
        or binding.get("code_bound") is not True
        or (complete_structure and cross_fit_audit.get("passed") is not True)
    )
    if success:
        status = "T10_2_1_SOURCE_COLLECTION_COMPLETE"
    elif provenance_invalid or terminal_reason not in resource_stop_reasons:
        status = "DATA_OR_PROVENANCE_INVALID"
    else:
        status = "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
    payload = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "phase": "collect",
        "status": status,
        "manifest_checksum": manifest["manifest_checksum"],
        "games": list(SOURCE_GAMES),
        "splits": {
            "discovery": list(DISCOVERY_SEEDS),
            "leave_one_game_out_confirmation": list(CONFIRMATION_SEEDS),
        },
        "event_count": len(events),
        "authorized_action_count": accounting.authorized_intent_count,
        "unresolved_intent_count": (
            accounting.explicitly_unresolved_intent_count
        ),
        "action_accounting": action_accounting,
        "durability": durability,
        "checks": checks,
        "events": _artifact_descriptor(ledger_path),
        "cross_fit_audit": _artifact_descriptor(cross_fit_path),
        "cross_fit_checks": dict(cross_fit_audit["checks"]),
        "timing": {
            "clock": "time.perf_counter",
            "current_invocation_started": started,
            "current_invocation_finished": finished,
            "current_invocation_seconds": finished - started,
            "cumulative_active_seconds": elapsed,
            "stop_new_actions_seconds": COLLECTION_COOPERATIVE_SECONDS,
            "absolute_seconds": COLLECTION_ABSOLUTE_SECONDS,
        },
        "resource_preflight": (
            None if snapshot_payload is None else dict(snapshot_payload)
        ),
        "terminal_reason": terminal_reason or None,
        "factory": binding,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    if data_error is not None:
        payload["error"] = {
            "kind": type(data_error).__name__,
            "message_sha256": canonical_sha256(str(data_error)),
        }
    if status == "SOURCE_ACQUISITION_OR_RESOURCE_MISS" and not (
        protocol._collection_failure_is_attested(payload)
        and protocol._collection_durable_evidence_matches(
            payload,
            manifest=manifest,
            output_dir=destination,
            require_invocation=False,
        )
    ):
        payload["status"] = "DATA_OR_PROVENANCE_INVALID"
        payload["error"] = {
            "kind": "UnattestedAcquisitionStop",
            "message_sha256": canonical_sha256(terminal_reason),
        }
    projected_report_core_checksum = canonical_sha256(payload)
    invocation_terminal = claim_terminal_receipt(
        elapsed,
        report_core_checksum=(
            projected_report_core_checksum
            if candidate_terminal_status == "CLOSED"
            else None
        ),
    )
    try:
        _t10_2_protocol.enforce_artifact_limit(
            invocation_terminal_path,
            kind="derived",
            limits=limits,
        )
    except Exception as exc:
        return publish_terminal_data(exc)
    if invocation_terminal.get("status") == "HARD_TIMEOUT":
        payload["checks"]["absolute_wall_bound"] = (
            _terminal_absolute_wall_bound(invocation_state, invocation_terminal)
        )
        payload["terminal_reason"] = "registered_collection_deadline"
        if payload["status"] == "T10_2_1_SOURCE_COLLECTION_COMPLETE":
            payload["status"] = "SOURCE_ACQUISITION_OR_RESOURCE_MISS"
        if payload["status"] == "SOURCE_ACQUISITION_OR_RESOURCE_MISS" and not (
            protocol._collection_failure_is_attested(payload)
            and protocol._collection_durable_evidence_matches(
                payload,
                manifest=manifest,
                output_dir=destination,
                require_invocation=False,
            )
        ):
            payload["status"] = "DATA_OR_PROVENANCE_INVALID"
            payload["error"] = {
                "kind": "UnattestedAcquisitionStop",
                "message_sha256": canonical_sha256(
                    "registered_collection_deadline"
                ),
            }
    payload["invocation"] = _invocation_report_binding(
        invocation_state,
        invocation_terminal,
    )
    if not protocol._collection_durable_evidence_matches(
        payload,
        manifest=manifest,
        output_dir=destination,
        require_invocation=True,
    ):
        return publish_terminal_data(
            JournalIntegrityError(
                "terminal report did not reconstruct with invocation receipts"
            )
        )
    report = _signed(payload, checksum_key="report_checksum")
    _validate_report_invocation_binding(
        report,
        opened=invocation_state,
        terminal=invocation_terminal,
        required=True,
    )
    encoded_size = len((canonical_json(report) + "\n").encode("utf-8"))
    if encoded_size > limits.maximum_derived_file_bytes:
        return publish_terminal_data(
            _t10_2_protocol.ResourceGateError(
                "collection report exceeds registered size"
            )
        )
    try:
        _write_once(report_path, report)
        _t10_2_protocol.enforce_artifact_limit(
            report_path,
            kind="derived",
            limits=limits,
        )
    finally:
        lease.release()
    return report


def _remaining_collection_watchdog_seconds(destination: Path) -> float:
    """Reconstruct active budget without charging process downtime."""

    state_path = destination / INVOCATION_STATE_FILENAME
    terminal_path = destination / INVOCATION_TERMINAL_FILENAME
    opened = _read_invocation_state(state_path)
    if opened is None:
        return COLLECTION_ABSOLUTE_SECONDS
    terminal = _read_invocation_terminal(terminal_path, opened=opened)
    if terminal is not None:
        # A terminal resume may only serialize an idempotent existing report or
        # a fail-closed DATA report.  This bounded recovery window authorizes no
        # environment/factory action and is not counted as collection time.
        return LANE_FINALIZATION_SECONDS
    active_seconds = float(opened["cumulative_active_seconds"])
    checkpoint_path = destination / CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        checkpoint = CollectionCheckpoint.from_dict(
            _read_canonical_json(checkpoint_path)
        )
        if checkpoint.manifest_checksum != opened["manifest_checksum"]:
            raise JournalIntegrityError("watchdog checkpoint escaped invocation")
        active_seconds = max(
            active_seconds,
            checkpoint.cumulative_active_seconds,
        )
        if checkpoint.open_lane_id is not None:
            active_seconds += min(
                RESET_HARD_SECONDS,
                max(
                    0.0,
                    LANE_HARD_SECONDS - checkpoint.open_lane_elapsed_seconds,
                ),
            )
    remaining = COLLECTION_ABSOLUTE_SECONDS - active_seconds
    if remaining > 0.0:
        return remaining
    _claim_invocation_terminal(
        terminal_path,
        opened=opened,
        status="HARD_TIMEOUT",
        cumulative_active_seconds=active_seconds,
        terminal_monotonic=time.monotonic(),
    )
    return LANE_FINALIZATION_SECONDS


def collect_phase(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None,
    env_factory: T10_2_1SourceFactory | None,
    resource_probe: Callable[[str | Path | None], Any] = (
        _t10_2_protocol.resource_snapshot
    ),
    limits: Any = _t10_2_protocol.DEFAULT_RESOURCE_LIMITS,
    clock: Callable[[], float] = time.perf_counter,
    process_context: Any = None,
) -> dict[str, Any]:
    """Run collection under a real 5,400-second external process-tree watchdog."""

    started = float(clock())
    watchdog_context = multiprocessing.get_context("spawn")
    cancel_event = watchdog_context.Event()
    registered_root = Path(repo_root or _t10_2_protocol._repo_root()).resolve()
    registered_destination = (registered_root / DEFAULT_OUTPUT_DIR).resolve()
    invocation_state_path = registered_destination / INVOCATION_STATE_FILENAME
    invocation_terminal_path = (
        registered_destination / INVOCATION_TERMINAL_FILENAME
    )
    try:
        watchdog_seconds = _remaining_collection_watchdog_seconds(
            registered_destination
        )
    except Exception:
        # Corrupt timing evidence gets only a bounded DATA-finalization window.
        watchdog_seconds = LANE_FINALIZATION_SECONDS
    deadline = time.monotonic() + watchdog_seconds
    watchdog = watchdog_context.Process(
        target=_collection_hard_watchdog_entry,
        args=(
            os.getpid(),
            deadline,
            cancel_event,
            str(invocation_state_path),
            str(invocation_terminal_path),
        ),
        name="sage-t10-2-1-collection-hard-watchdog",
        daemon=True,
    )
    try:
        watchdog.start()
    except Exception as exc:
        raise WorkerProtocolError(
            "external collection hard watchdog could not start"
        ) from exc
    result: dict[str, Any]
    try:
        result = _collect_phase_impl(
            manifest_path=manifest_path,
            output_dir=output_dir,
            repo_root=repo_root,
            env_factory=env_factory,
            resource_probe=resource_probe,
            limits=limits,
            clock=clock,
            process_context=process_context,
            _started_override=started,
        )
    finally:
        cancel_event.set()
        watchdog.join(timeout=2.0)
        if watchdog.is_alive():
            watchdog.terminate()
            watchdog.join(timeout=2.0)
        if watchdog.is_alive():
            kill = getattr(watchdog, "kill", None)
            if callable(kill):
                kill()
                watchdog.join(timeout=2.0)
    committed = _read_existing_collection_report(
        registered_destination / COLLECTION_REPORT_FILENAME
    )
    if committed is None or committed != result:
        raise JournalIntegrityError("collection return escaped its terminal commit")
    is_minimal_data = bool(
        committed.get("status") == "DATA_OR_PROVENANCE_INVALID"
        and _mapping(committed.get("events", {})).get("available") is False
    )
    if is_minimal_data and committed.get("invocation") is None:
        return committed
    opened = _read_invocation_state(invocation_state_path)
    terminal = _read_invocation_terminal(
        invocation_terminal_path,
        opened=opened,
    )
    _validate_report_invocation_binding(
        committed,
        opened=opened,
        terminal=terminal,
        required=not is_minimal_data,
    )
    return committed


__all__ = [
    "ActionIntent",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_FORMAT_VERSION",
    "COLLECTION_ABSOLUTE_SECONDS",
    "COLLECTION_COOPERATIVE_SECONDS",
    "COLLECTION_FORMAT_VERSION",
    "CollectionCheckpoint",
    "CONFIRMATION_SEEDS",
    "DISCOVERY_SEEDS",
    "DurableCollectionJournal",
    "FIT_SEED",
    "FORMAT_VERSION",
    "GaugeDecisionEngine",
    "GaugeProgramPosterior",
    "JournalConflictError",
    "JournalIntegrityError",
    "JournalAccounting",
    "LANE_FINALIZATION_SECONDS",
    "LANE_HARD_SECONDS",
    "LaneReport",
    "PERMUTATION_SEED",
    "PhysicalEventBundle",
    "PhysicalEventReceipt",
    "PosteriorUpdateReceipt",
    "ProcessResetWatchdog",
    "RESET_COOPERATIVE_SECONDS",
    "RESET_HARD_SECONDS",
    "ResetReport",
    "ResetWatchdog",
    "ResetWorkSpec",
    "ResumeRefusalError",
    "SOURCE_ACTIONS_PER_RESET",
    "SOURCE_GAMES",
    "SOURCE_MAXIMUM_AUTHORIZED_INTENTS",
    "SOURCE_RESETS_PER_LANE",
    "SourceLaneKey",
    "T10_2_1SourceFactory",
    "UnresolvedIntentReceipt",
    "WorkerOutcome",
    "WorkerProtocolError",
    "WorkerTimeoutError",
    "canonical_json",
    "canonical_sha256",
    "collect_phase",
    "confirmation_controller_order",
    "generate_mixed_grammar",
    "rank_option_sequence_signatures",
    "repeat",
    "reset_work_specs",
    "source_lane_registry",
]
