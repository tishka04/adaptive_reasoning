"""Readiness-instrumented durable runtime for SAGE.T10.2.2.

T10.2.2 is an orchestration-only amendment.  The scientific kernel (T10.2) and
the durable, spawn-per-reset acquisition/persistence layer (T10.2.1) are frozen
and imported verbatim; nothing in ``t10_2_1_runtime`` is modified.

The single structural change is that the parent process now *observes* every
reset it supervises without altering the child, the journal, or the frozen
reset/lane/checkpoint schemas.  The T10.2.1 factory already exposes an
injectable ``watchdog`` (see ``T10_2_1SourceFactory.__init__``) that stays on the
parent and is deliberately dropped when the factory is pickled to the child.
``LaneTimingWatchdog`` uses that seam to wrap the parent-side message handler and
record, per reset:

* controller readiness  (the moment the frozen worker begins ``open`` -- the
  posterior/controller is already fitted at that point);
* environment readiness (the moment ``reset`` finishes -- the environment is
  opened and reset);
* the first authorized intent (first ``action_intent`` message); and
* the first committed transition (first ``physical_event`` the parent seals).

From those, T10.2.2 derives, in a *sidecar* report that binds the frozen
checkpoint's exact revision and checksum:

* item 2 -- lane-start-to-first-committed startup latency, kept out of the
  interaction budget;
* item 3 -- a readiness-relative first-intent deadline;
* item 4 -- an interaction deadline anchored at ``max(controller, environment)``
  readiness;
* items 1/5/6/7 -- the checkpoint binding, evidence funnel, schema-family
  partition, and induction canary from :mod:`theory.sage_t.t10_2_2_protocol`.

The sidecar is written next to the frozen artifacts and never mutates them.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _t10_2_1
from . import t10_2_1_runtime as _t10_2_1_runtime
from . import t10_2_2_protocol as _protocol

FORMAT_VERSION = "sage-t10.2.2-runtime-v1"
T10_2_2_COLLECTION_REPORT_FORMAT_VERSION = "sage-t10.2.2-collection-report-v1"
T10_2_2_COLLECTION_REPORT_FILENAME = "t10_2_2_collection_report.json"

# The interaction budget is charged only from joint readiness (item 4); it
# mirrors the frozen cooperative reset budget so the science is unchanged.
INTERACTION_BUDGET_SECONDS = _t10_2_1_runtime.RESET_COOPERATIVE_SECONDS
# The first authorized intent must arrive within this readiness-relative window
# (item 3).  It is strictly smaller than the interaction budget.
FIRST_INTENT_BUDGET_SECONDS = 30.0

CHECKPOINT_FILENAME = _t10_2_1_runtime.CHECKPOINT_FILENAME
COLLECTION_REPORT_FILENAME = _t10_2_1_runtime.COLLECTION_REPORT_FILENAME

canonical_sha256 = _t10_2_1.canonical_sha256
signed_payload = _t10_2_1.signed_payload
write_compact_json = _t10_2_1.write_compact_json
_read_signed_json = _t10_2_1._read_signed_json
ManifestDriftError = _t10_2_1.ManifestDriftError
ProtocolError = _t10_2_1.ProtocolError


# ---------------------------------------------------------------------------
# Per-reset timing record captured by the parent observer.
# ---------------------------------------------------------------------------
@dataclass
class ResetTiming:
    work_id: str
    lane: dict[str, Any]
    reset_index: int
    reset_started_at: float | None = None
    controller_ready_at: float | None = None
    environment_ready_at: float | None = None
    first_intent_at: float | None = None
    first_committed_at: float | None = None
    finished_at: float | None = None
    status: str | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "lane": dict(self.lane),
            "reset_index": self.reset_index,
            "reset_started_at": self.reset_started_at,
            "controller_ready_at": self.controller_ready_at,
            "environment_ready_at": self.environment_ready_at,
            "first_intent_at": self.first_intent_at,
            "first_committed_at": self.first_committed_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "stop_reason": self.stop_reason,
        }


class CollectionTimingRecorder:
    """Accumulates parent-observed timings across every supervised reset."""

    def __init__(self) -> None:
        self._resets: dict[str, ResetTiming] = {}
        self._order: list[str] = []

    def _entry(self, work: Any) -> ResetTiming:
        work_id = str(work.work_id)
        if work_id not in self._resets:
            self._resets[work_id] = ResetTiming(
                work_id=work_id,
                lane=dict(work.lane.to_dict()),
                reset_index=int(work.reset_index),
            )
            self._order.append(work_id)
        return self._resets[work_id]

    def note_reset_started(self, work: Any, at: float) -> None:
        entry = self._entry(work)
        if entry.reset_started_at is None:
            entry.reset_started_at = float(at)

    def observe(self, work: Any, message: Mapping[str, Any], at: float) -> None:
        """Record a single parent-observed worker message timestamp."""

        entry = self._entry(work)
        kind = str(message.get("kind", ""))
        if kind == "reset_operation":
            payload = message.get("payload") or {}
            operation = str(payload.get("operation", ""))
            stage = str(payload.get("stage", ""))
            if (
                operation == "open"
                and stage == "started"
                and entry.controller_ready_at is None
            ):
                # The frozen worker fits the controller/posterior before ``open``.
                entry.controller_ready_at = float(at)
            elif operation == "reset" and stage == "finished":
                entry.environment_ready_at = float(at)
        elif kind == "action_intent" and entry.first_intent_at is None:
            entry.first_intent_at = float(at)
        elif kind == "physical_event" and entry.first_committed_at is None:
            entry.first_committed_at = float(at)

    def note_reset_finished(
        self, work: Any, at: float, *, status: str, stop_reason: str | None
    ) -> None:
        entry = self._entry(work)
        entry.finished_at = float(at)
        entry.status = status
        entry.stop_reason = stop_reason

    def reset_timings(self) -> tuple[ResetTiming, ...]:
        return tuple(self._resets[work_id] for work_id in self._order)

    def lane_windows(self) -> list[dict[str, Any]]:
        """Aggregate reset timings into one start/first-committed/finish window."""

        lanes: dict[str, dict[str, Any]] = {}
        for timing in self.reset_timings():
            lane_id = str(timing.lane.get("lane_id"))
            window = lanes.setdefault(
                lane_id,
                {
                    "lane": dict(timing.lane),
                    "lane_started_seconds": None,
                    "first_committed_transition_seconds": None,
                    "lane_finished_seconds": None,
                },
            )
            window["lane_started_seconds"] = _min_opt(
                window["lane_started_seconds"], timing.reset_started_at
            )
            window["first_committed_transition_seconds"] = _min_opt(
                window["first_committed_transition_seconds"], timing.first_committed_at
            )
            window["lane_finished_seconds"] = _max_opt(
                window["lane_finished_seconds"], timing.finished_at
            )
        return list(lanes.values())


def _min_opt(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return float(candidate)
    return min(current, float(candidate))


def _max_opt(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return float(candidate)
    return max(current, float(candidate))


# ---------------------------------------------------------------------------
# Parent-side supervisor that observes and delegates.
# ---------------------------------------------------------------------------
class LaneTimingWatchdog:
    """Wraps a frozen ``ProcessResetWatchdog`` to observe parent messages.

    It never changes supervision semantics: every message is forwarded to the
    frozen handler unchanged and the frozen watchdog owns all timeout authority.
    """

    def __init__(
        self,
        *,
        recorder: CollectionTimingRecorder,
        clock: Callable[[], float] = time.perf_counter,
        inner: Any | None = None,
    ) -> None:
        self._recorder = recorder
        self._clock = clock
        self._inner = inner or _t10_2_1_runtime.ProcessResetWatchdog(clock=clock)

    def supervise(
        self,
        process: Any,
        *,
        work: Any,
        cancel_event: Any,
        outbound_queue: Any,
        inbound_queue: Any,
        message_handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        cooperative_seconds: float,
        hard_seconds: float,
        started_at: float,
    ) -> Any:
        self._recorder.note_reset_started(work, started_at)

        def observing_handler(
            message: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            # Observe first so a message that trips the intent deadline (and
            # raises inside the frozen handler) is still recorded as attempted.
            self._recorder.observe(work, message, float(self._clock()))
            return message_handler(message)

        status = "EXCEPTION"
        stop_reason: str | None = None
        try:
            outcome = self._inner.supervise(
                process,
                work=work,
                cancel_event=cancel_event,
                outbound_queue=outbound_queue,
                inbound_queue=inbound_queue,
                message_handler=observing_handler,
                cooperative_seconds=cooperative_seconds,
                hard_seconds=hard_seconds,
                started_at=started_at,
            )
            status = str(getattr(outcome, "status", "UNKNOWN"))
            payload = getattr(outcome, "payload", None) or {}
            stop_reason = payload.get("stop_reason")
            return outcome
        finally:
            self._recorder.note_reset_finished(
                work,
                float(self._clock()),
                status=status,
                stop_reason=stop_reason,
            )


def timing_source_factory(
    *,
    manifest: Mapping[str, Any],
    recorder: CollectionTimingRecorder,
    clock: Callable[[], float] = time.perf_counter,
) -> _t10_2_1_runtime.T10_2_1SourceFactory:
    """A frozen T10.2.1 factory with a T10.2.2 timing watchdog attached."""

    return _t10_2_1_runtime.T10_2_1SourceFactory(
        manifest=manifest,
        watchdog=LaneTimingWatchdog(recorder=recorder, clock=clock),
    )


# ---------------------------------------------------------------------------
# Sidecar report construction (pure; unit-testable without multiprocessing).
# ---------------------------------------------------------------------------
def _iter_reset_reports(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    resets: list[dict[str, Any]] = []
    for lane in checkpoint.get("lane_reports", []) or []:
        if not isinstance(lane, Mapping):
            continue
        for reset in lane.get("resets", []) or []:
            if isinstance(reset, Mapping):
                resets.append(dict(reset))
    return resets


def _parse_schema_key(key: Any) -> tuple[str, int]:
    if isinstance(key, (list, tuple)) and len(key) == 2:
        return str(key[0]), int(key[1])
    try:
        parsed = json.loads(str(key))
    except json.JSONDecodeError:
        return str(key), 0
    if isinstance(parsed, list) and len(parsed) == 2:
        return str(parsed[0]), int(parsed[1])
    return str(key), 0


def _aggregate_schema_counts(
    reset_reports: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int], dict[str, int]]:
    learned: Counter[tuple[str, int]] = Counter()
    independent: Counter[tuple[str, int]] = Counter()
    grounding: Counter[str] = Counter()
    for report in reset_reports:
        continuation = report.get("continuation")
        if not isinstance(continuation, Mapping):
            continue
        for key, value in (continuation.get("learned_schema_counts") or {}).items():
            learned[_parse_schema_key(key)] += int(value)
        for key, value in (continuation.get("independent_schema_counts") or {}).items():
            independent[_parse_schema_key(key)] += int(value)
        for key, value in (continuation.get("grounding_counts") or {}).items():
            grounding[str(key)] += int(value)
    return dict(learned), dict(independent), dict(grounding)


def build_t10_2_2_collection_report(
    *,
    recorder: CollectionTimingRecorder,
    checkpoint: Mapping[str, Any],
    collection_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the readiness-anchored T10.2.2 sidecar over frozen artifacts."""

    manifest_checksum = collection_report.get("manifest_checksum")
    checkpoint_binding = _protocol.build_checkpoint_binding(
        collection_report=collection_report, checkpoint=checkpoint
    )

    # Item 2: lane startup timing from the observed windows.
    lane_timings: list[dict[str, Any]] = []
    for window in recorder.lane_windows():
        started = window["lane_started_seconds"]
        finished = window["lane_finished_seconds"]
        if started is None or finished is None:
            continue
        lane_timings.append(
            _protocol.compute_lane_startup_timing(
                lane=window["lane"],
                lane_started_seconds=started,
                first_committed_transition_seconds=window[
                    "first_committed_transition_seconds"
                ],
                lane_finished_seconds=finished,
            )
        )
    phase_timing = _protocol.build_phase_timing(lane_timings=lane_timings)

    # Items 3 and 4: readiness gates and readiness-relative first-intent status.
    readiness_gates: list[dict[str, Any]] = []
    first_intent_statuses: list[dict[str, Any]] = []
    for timing in recorder.reset_timings():
        if timing.controller_ready_at is None or timing.environment_ready_at is None:
            first_intent_statuses.append(
                {
                    "work_id": timing.work_id,
                    "readiness_observed": False,
                    "status": None,
                }
            )
            continue
        readiness_gates.append(
            _protocol.readiness_gate(
                lane=timing.lane,
                controller_ready_at=timing.controller_ready_at,
                environment_ready_at=timing.environment_ready_at,
                interaction_budget_seconds=INTERACTION_BUDGET_SECONDS,
            )
        )
        status = _protocol.classify_first_intent(
            controller_ready_at=timing.controller_ready_at,
            environment_ready_at=timing.environment_ready_at,
            first_intent_authorized_at=timing.first_intent_at,
            first_intent_budget_seconds=FIRST_INTENT_BUDGET_SECONDS,
        )
        first_intent_statuses.append(
            {
                "work_id": timing.work_id,
                "readiness_observed": True,
                "status": status,
            }
        )

    # Item 5: evidence funnel from the frozen reset reports.
    reset_reports = _iter_reset_reports(checkpoint)
    evidence_funnel = _protocol.evidence_funnel_from_reset_reports(reset_reports)

    # Item 6: canonical schema families vs grounded instances.
    learned, independent, grounding = _aggregate_schema_counts(reset_reports)
    schema_evidence = _protocol.partition_schema_evidence(
        learned_schema_counts=learned,
        independent_schema_counts=independent,
        grounding_counts=grounding,
    )

    # Item 7: controlled end-to-end induction canary.
    induction_canary = _protocol.run_induction_canary()

    return signed_payload(
        {
            "format_version": T10_2_2_COLLECTION_REPORT_FORMAT_VERSION,
            "phase": "collect",
            "manifest_checksum": manifest_checksum,
            "frozen_collection_report_checksum": collection_report.get(
                "report_checksum"
            ),
            "checkpoint_binding": checkpoint_binding,
            "phase_timing": phase_timing,
            "readiness_gates": readiness_gates,
            "first_intent": {
                "budget_seconds": FIRST_INTENT_BUDGET_SECONDS,
                "per_reset": first_intent_statuses,
                "any_timeout": any(
                    row.get("status") == "first_intent_timeout"
                    for row in first_intent_statuses
                ),
            },
            "interaction_budget_seconds": INTERACTION_BUDGET_SECONDS,
            "evidence_funnel": evidence_funnel,
            "schema_evidence": schema_evidence,
            "induction_canary": induction_canary,
            "reset_timings": [timing.to_dict() for timing in recorder.reset_timings()],
        },
        checksum_key="report_checksum",
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestDriftError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestDriftError(f"{label} is not a JSON object: {path}")
    return dict(payload)


def write_t10_2_2_collection_report(
    *,
    recorder: CollectionTimingRecorder,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Read the frozen collection report + checkpoint and write the sidecar."""

    destination = Path(output_dir)
    collection_report = _read_signed_json(
        destination / COLLECTION_REPORT_FILENAME, checksum_key="report_checksum"
    )
    checkpoint = _load_json_object(
        destination / CHECKPOINT_FILENAME, label="collection checkpoint"
    )
    report = build_t10_2_2_collection_report(
        recorder=recorder,
        checkpoint=checkpoint,
        collection_report=collection_report,
    )
    write_compact_json(destination / T10_2_2_COLLECTION_REPORT_FILENAME, report)
    return report


# ---------------------------------------------------------------------------
# Collect phase: run the frozen collection with a timing watchdog, then sidecar.
# ---------------------------------------------------------------------------
def collect_phase(
    *,
    manifest_path: str | Path = _t10_2_1.DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = _t10_2_1_runtime.DEFAULT_OUTPUT_DIR,
    repo_root: str | Path | None = None,
    env_factory: _t10_2_1_runtime.T10_2_1SourceFactory | None = None,
    recorder: CollectionTimingRecorder | None = None,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run frozen T10.2.1 collection instrumented for T10.2.2, then bind timing.

    The frozen ``collect_phase`` writes ``collection_report.json`` and
    ``source_collection_checkpoint.json`` into the manifest-registered artifact
    directory.  T10.2.2 then writes ``t10_2_2_collection_report.json`` beside
    them, binding the checkpoint's exact revision and checksum.
    """

    manifest = _t10_2_1.load_manifest(manifest_path, repo_root=repo_root)
    recorder = recorder or CollectionTimingRecorder()
    if env_factory is None:
        env_factory = timing_source_factory(
            manifest=manifest, recorder=recorder, clock=clock
        )
    elif env_factory.watchdog is None:
        env_factory.watchdog = LaneTimingWatchdog(recorder=recorder, clock=clock)

    _t10_2_1.collect_phase(
        manifest_path=manifest_path,
        output_dir=output_dir,
        repo_root=repo_root,
        env_factory=env_factory,
        clock=clock,
        **kwargs,
    )

    registered_root = Path(repo_root or _t10_2_1._repo_root()).resolve()
    registered_destination = (
        registered_root / _t10_2_1_runtime.DEFAULT_OUTPUT_DIR
    ).resolve()
    return write_t10_2_2_collection_report(
        recorder=recorder, output_dir=registered_destination
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect",))
    parser.add_argument("--manifest", default=str(_t10_2_1.DEFAULT_MANIFEST_PATH))
    parser.add_argument(
        "--output-dir", default=str(_t10_2_1_runtime.DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument("--repo-root", default=str(_t10_2_1._repo_root()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _t10_2_1.load_manifest(args.manifest, repo_root=args.repo_root)
    recorder = CollectionTimingRecorder()
    factory = timing_source_factory(manifest=manifest, recorder=recorder)
    try:
        payload = collect_phase(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            env_factory=factory,
            recorder=recorder,
        )
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(
            _t10_2_1.canonical_json(
                {"error": f"{type(exc).__name__}:{exc}", "phase": args.phase}
            )
        )
        return 2
    print(_t10_2_1.canonical_json(payload))
    return 0


__all__ = [
    "CollectionTimingRecorder",
    "FIRST_INTENT_BUDGET_SECONDS",
    "FORMAT_VERSION",
    "INTERACTION_BUDGET_SECONDS",
    "LaneTimingWatchdog",
    "ResetTiming",
    "T10_2_2_COLLECTION_REPORT_FILENAME",
    "T10_2_2_COLLECTION_REPORT_FORMAT_VERSION",
    "build_parser",
    "build_t10_2_2_collection_report",
    "collect_phase",
    "main",
    "timing_source_factory",
    "write_t10_2_2_collection_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
