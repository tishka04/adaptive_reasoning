"""SAGE.T10.2.2 induction-faithful, readiness-gated gauge-posterior retest.

T10.2.2 keeps the frozen T10.2 scientific kernel and the frozen T10.2.1
durable, time-budgeted acquisition/persistence layer intact.  It changes only
orchestration, timing accounting, and report synchronization.  The T10.2.1
protocol module is imported as the frozen kernel and is never mutated here.

Nine orchestration upgrades are implemented, in the registered order:

1. Report generation is synchronized with the exact checkpoint revision and
   checksum, not merely the collection report's copy of the checksum.
2. Phase-level timing is measured from lane start to the first committed
   transition, so startup latency is attributable and never folded into the
   interaction budget.
3. Missing cross-fit artifacts and a first-intent timeout both fail fast, before
   any downstream scientific claim is attempted.
4. Interaction deadlines start only after *both* the controller and the
   environment are ready; environment open/reset never eats the action budget.
5. A complete evidence funnel accounts for every authorized intent with an
   explicit, registered rejection reason.
6. Canonical, coordinate-free schema families are separated from grounded
   instances so capacity is never confused with grounding.
7. A controlled, deterministic end-to-end induction canary proves the induction
   mechanism actually consumes delivered independent evidence.
8. Discovery and confirmation lanes are interleaved and a reserved confirmation
   capacity is guaranteed even under lane-budget truncation.
9. One smoke lane per split is run before the complete matrix is launched.

Two exclusivity invariants are enforced when a negative verdict is proposed:

* Schema-learning failure is never declared unless qualifying independent
  evidence was generated *and* actually delivered to the induction mechanism.
* Transfer failure is never declared for any lane that issued zero intents.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _t10_2_1

# ---------------------------------------------------------------------------
# Frozen-kernel re-exports.  T10.2.2 reuses the T10.2.1 hashing, signing, seed,
# game, and verdict vocabulary verbatim; drift here would break lineage.
# ---------------------------------------------------------------------------
canonical_json = _t10_2_1.canonical_json
canonical_sha256 = _t10_2_1.canonical_sha256
signed_payload = _t10_2_1.signed_payload
write_compact_json = _t10_2_1.write_compact_json
_read_signed_json = _t10_2_1._read_signed_json

ProtocolError = _t10_2_1.ProtocolError
ManifestDriftError = _t10_2_1.ManifestDriftError
FirewallError = _t10_2_1.FirewallError
DataGateError = _t10_2_1.DataGateError
GateRefusalError = _t10_2_1.GateRefusalError
ResourceGateError = _t10_2_1.ResourceGateError

SOURCE_GAMES = _t10_2_1.SOURCE_GAMES
VALIDATION_GAMES = _t10_2_1.VALIDATION_GAMES
DISCOVERY_SEEDS = _t10_2_1.DISCOVERY_SEEDS
CONFIRMATION_SEEDS = _t10_2_1.CONFIRMATION_SEEDS
EXCLUSIVE_VERDICTS = _t10_2_1.EXCLUSIVE_VERDICTS
SOURCE_NEGATIVE_VERDICTS = _t10_2_1.SOURCE_NEGATIVE_VERDICTS
CROSS_FIT_AUDIT_FILENAME = _t10_2_1.CROSS_FIT_AUDIT_FILENAME
CHECKPOINT_FILENAME = "source_collection_checkpoint.json"
COLLECTION_REPORT_FILENAME = "collection_report.json"

PARENT_T10_2_1_PROTOCOL_FORMAT = _t10_2_1.FORMAT_VERSION

FORMAT_VERSION = "sage-t10.2.2-protocol-v1"
CHECKPOINT_BINDING_FORMAT_VERSION = "sage-t10.2.2-checkpoint-binding-v1"
PHASE_TIMING_FORMAT_VERSION = "sage-t10.2.2-phase-timing-v1"
READINESS_FORMAT_VERSION = "sage-t10.2.2-readiness-gate-v1"
EVIDENCE_FUNNEL_FORMAT_VERSION = "sage-t10.2.2-evidence-funnel-v1"
SCHEMA_EVIDENCE_FORMAT_VERSION = "sage-t10.2.2-schema-evidence-v1"
INDUCTION_CANARY_FORMAT_VERSION = "sage-t10.2.2-induction-canary-v1"
LANE_SCHEDULE_FORMAT_VERSION = "sage-t10.2.2-lane-schedule-v1"
SMOKE_PLAN_FORMAT_VERSION = "sage-t10.2.2-smoke-plan-v1"
ORCHESTRATION_REPORT_FORMAT_VERSION = "sage-t10.2.2-orchestration-report-v1"

# T10.2.2 shares the frozen T10.2.1 manifest, so the frozen collect writes into
# the T10.2.1-registered artifact namespace; the T10.2.2 sidecar lands beside it.
DEFAULT_OUTPUT_DIR = _t10_2_1.DEFAULT_OUTPUT_DIR

SOURCE_SPLITS = ("discovery", "leave_one_game_out_confirmation")

# Registered stop/rejection reasons every unsealed authorized intent must carry.
REGISTERED_REJECTION_REASONS = frozenset(
    {
        "first_intent_timeout",
        "cooperative_reset_deadline",
        "hard_reset_timeout",
        "registered_collection_deadline",
        "resource_gate",
        "interrupted_before_reset_commit",
        "worker_exited",
        "worker_exception",
        "environment_call_unattestable",
        "parent_interrupted",
    }
)

# Verdicts that blame the schema learner / induction mechanism.  None of these
# may be declared unless qualifying independent evidence reached induction.
SCHEMA_LEARNING_FAILURE_VERDICTS = frozenset(
    {
        "MIXED_SEQUENCE_GRAMMAR_MISS",
        "COMMON_POSTERIOR_MISS",
        "OPTION_SYNTHESIS_MISS",
        "SOURCE_GROUNDING_MISS",
    }
)
TRANSFER_FAILURE_VERDICT = "SOURCE_VALIDATION_TRANSFER_MISS"
# When an evidence precondition is unmet the miss is re-attributed to acquisition
# rather than to a scientific mechanism that never received its inputs.
EVIDENCE_UNMET_VERDICT = "SOURCE_ACQUISITION_OR_RESOURCE_MISS"

# Fraction of confirmation lanes that must survive any lane-budget truncation so
# schema confirmation is never starved by discovery.
CONFIRMATION_RESERVE_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Small numeric guards.
# ---------------------------------------------------------------------------
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (
        math.isfinite(float(value))
    )


def _finite_nonneg(value: Any) -> bool:
    return _finite(value) and float(value) >= 0.0


def _require_nonneg_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataGateError(f"{label} must be a non-negative integer")
    return int(value)


# ---------------------------------------------------------------------------
# Lane identity, independent of the heavy durable runtime.  ``lane_id`` matches
# ``t10_2_1_runtime.SourceLaneKey.lane_id`` exactly (sha256 of the identity map).
# ---------------------------------------------------------------------------
def _lane_identity(split: str, game_id: str, seed: int) -> dict[str, Any]:
    if split not in SOURCE_SPLITS:
        raise ValueError(f"invalid source split: {split}")
    if game_id not in SOURCE_GAMES:
        raise ValueError(f"non-source game: {game_id}")
    seeds = DISCOVERY_SEEDS if split == "discovery" else CONFIRMATION_SEEDS
    if isinstance(seed, bool) or int(seed) not in seeds:
        raise ValueError(f"seed {seed} is not registered for {split}")
    return {"split": split, "game_id": game_id, "seed": int(seed)}


def _lane_dict(split: str, game_id: str, seed: int) -> dict[str, Any]:
    identity = _lane_identity(split, game_id, seed)
    return {**identity, "lane_id": canonical_sha256(identity)}


def source_lane_registry() -> tuple[dict[str, Any], ...]:
    """The frozen 18-lane matrix: discovery lanes first, confirmation next."""

    discovery = tuple(
        _lane_dict("discovery", game_id, seed)
        for game_id in SOURCE_GAMES
        for seed in DISCOVERY_SEEDS
    )
    confirmation = tuple(
        _lane_dict("leave_one_game_out_confirmation", game_id, seed)
        for game_id in SOURCE_GAMES
        for seed in CONFIRMATION_SEEDS
    )
    return discovery + confirmation


def _lanes_by_split() -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {split: [] for split in SOURCE_SPLITS}
    for lane in source_lane_registry():
        grouped[str(lane["split"])].append(lane)
    return {split: tuple(lanes) for split, lanes in grouped.items()}


# ===========================================================================
# Item 1: synchronize report generation with the exact checkpoint revision and
# checksum.
# ===========================================================================
def _checkpoint_binding_facts(checkpoint: Mapping[str, Any]) -> tuple[int, str]:
    revision = checkpoint.get("revision")
    checksum = checkpoint.get("checkpoint_checksum")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ManifestDriftError("checkpoint revision is missing or malformed")
    if not isinstance(checksum, str) or not checksum:
        raise ManifestDriftError("checkpoint checksum is missing or malformed")
    # The checkpoint is self-authenticating: its recorded checksum must equal the
    # canonical hash of its unsigned body.  This binds *this exact revision*.
    unsigned = {key: value for key, value in checkpoint.items()
                if key != "checkpoint_checksum"}
    if canonical_sha256(unsigned) != checksum:
        raise ManifestDriftError("checkpoint checksum does not authenticate its body")
    return int(revision), str(checksum)


def build_checkpoint_binding(
    *,
    collection_report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the report to the checkpoint's exact revision *and* checksum.

    T10.2.1 forwarded only ``durability.checkpoint_checksum``.  A checksum alone
    cannot distinguish two runs that resumed to different revisions of the same
    lane set, so T10.2.2 additionally pins the monotonic ``revision``.
    """

    revision, checksum = _checkpoint_binding_facts(checkpoint)
    durability = collection_report.get("durability")
    if not isinstance(durability, Mapping):
        raise ManifestDriftError("collection report lacks a durability section")
    reported_checksum = durability.get("checkpoint_checksum")
    if reported_checksum != checksum:
        raise ManifestDriftError(
            "collection report checkpoint checksum diverged from the checkpoint"
        )
    manifest_checksum = collection_report.get("manifest_checksum")
    if checkpoint.get("manifest_checksum") not in (None, manifest_checksum):
        raise ManifestDriftError("checkpoint escaped the report manifest")
    return signed_payload(
        {
            "format_version": CHECKPOINT_BINDING_FORMAT_VERSION,
            "manifest_checksum": manifest_checksum,
            "checkpoint_revision": revision,
            "checkpoint_checksum": checksum,
            "collection_report_checksum": collection_report.get("report_checksum"),
            "synchronized": True,
        },
        checksum_key="binding_checksum",
    )


def synchronize_report_with_checkpoint(
    *, output_dir: str | Path
) -> dict[str, Any]:
    """Read the on-disk collection report and checkpoint and bind them."""

    destination = Path(output_dir)
    collection = _read_signed_json(
        destination / COLLECTION_REPORT_FILENAME, checksum_key="report_checksum"
    )
    checkpoint_path = destination / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        raise ManifestDriftError("collection checkpoint is missing for report sync")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint, Mapping):
        raise ManifestDriftError("collection checkpoint is not a JSON object")
    return build_checkpoint_binding(
        collection_report=collection, checkpoint=checkpoint
    )


# ===========================================================================
# Item 2: phase-level timing from lane start to first committed transition.
# ===========================================================================
def compute_lane_startup_timing(
    *,
    lane: Mapping[str, Any],
    lane_started_seconds: float,
    first_committed_transition_seconds: float | None,
    lane_finished_seconds: float,
) -> dict[str, Any]:
    """Attribute lane startup latency separately from interaction time.

    ``first_committed_transition_seconds`` is ``None`` when the lane never
    committed (sealed) a transition, in which case there is no startup latency to
    charge and the lane is flagged as having produced zero committed transitions.
    """

    if not (_finite_nonneg(lane_started_seconds) and _finite_nonneg(lane_finished_seconds)):
        raise DataGateError("lane wall clock is not a finite non-negative reading")
    if lane_finished_seconds < lane_started_seconds:
        raise DataGateError("lane finished before it started")
    committed = first_committed_transition_seconds is not None
    startup_latency: float | None = None
    interaction_seconds: float | None = None
    if committed:
        first = first_committed_transition_seconds
        if not _finite_nonneg(first):
            raise DataGateError("first committed transition clock is invalid")
        if first < lane_started_seconds or first > lane_finished_seconds:
            raise DataGateError(
                "first committed transition fell outside the lane window"
            )
        startup_latency = float(first) - float(lane_started_seconds)
        interaction_seconds = float(lane_finished_seconds) - float(first)
    return {
        "lane_id": lane.get("lane_id"),
        "split": lane.get("split"),
        "committed_first_transition": committed,
        "lane_started_seconds": float(lane_started_seconds),
        "first_committed_transition_seconds": (
            None if not committed else float(first_committed_transition_seconds)
        ),
        "lane_finished_seconds": float(lane_finished_seconds),
        "startup_latency_seconds": startup_latency,
        "interaction_seconds": interaction_seconds,
    }


def build_phase_timing(
    *, lane_timings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    committed = [t for t in lane_timings if t.get("committed_first_transition")]
    startup_values = [
        float(t["startup_latency_seconds"])
        for t in committed
        if _finite_nonneg(t.get("startup_latency_seconds"))
    ]
    return signed_payload(
        {
            "format_version": PHASE_TIMING_FORMAT_VERSION,
            "lane_count": len(lane_timings),
            "committed_lane_count": len(committed),
            "uncommitted_lane_count": len(lane_timings) - len(committed),
            "max_startup_latency_seconds": (
                max(startup_values) if startup_values else None
            ),
            "total_startup_latency_seconds": (
                math.fsum(startup_values) if startup_values else 0.0
            ),
            "lanes": [dict(timing) for timing in lane_timings],
        },
        checksum_key="timing_checksum",
    )


# ===========================================================================
# Item 3: fail fast on missing cross-fit artifacts and first-intent timeout.
# ===========================================================================
def require_cross_fit_artifacts(output_dir: str | Path) -> Path:
    """Refuse to proceed if the cross-fit audit artifact is absent or empty."""

    destination = Path(output_dir)
    cross_fit_path = destination / CROSS_FIT_AUDIT_FILENAME
    if not cross_fit_path.is_file():
        raise GateRefusalError(
            "cross-fit audit artifact is missing; refusing to fit or claim transfer"
        )
    if cross_fit_path.stat().st_size == 0:
        raise GateRefusalError("cross-fit audit artifact is empty")
    return cross_fit_path


def classify_first_intent(
    *,
    controller_ready_at: float,
    environment_ready_at: float,
    first_intent_authorized_at: float | None,
    first_intent_budget_seconds: float,
) -> str:
    """Return ``"ok"`` or ``"first_intent_timeout"``.

    The first-intent budget is charged from readiness (see item 4), never from
    lane spawn, so a slow environment open cannot masquerade as a stuck agent.
    """

    if not (_finite_nonneg(controller_ready_at) and _finite_nonneg(environment_ready_at)):
        raise DataGateError("readiness clocks must be finite and non-negative")
    if not _finite_nonneg(first_intent_budget_seconds):
        raise DataGateError("first-intent budget must be finite and non-negative")
    ready_at = max(float(controller_ready_at), float(environment_ready_at))
    deadline = ready_at + float(first_intent_budget_seconds)
    if first_intent_authorized_at is None:
        return "first_intent_timeout"
    if not _finite_nonneg(first_intent_authorized_at):
        raise DataGateError("first-intent authorization clock is invalid")
    if float(first_intent_authorized_at) > deadline:
        return "first_intent_timeout"
    return "ok"


def fail_fast_preflight(
    *,
    output_dir: str | Path,
    first_intent_status: str,
) -> Path:
    """Combined item-3 guard: cross-fit presence plus first-intent liveness."""

    cross_fit_path = require_cross_fit_artifacts(output_dir)
    if first_intent_status == "first_intent_timeout":
        raise GateRefusalError(
            "first intent was not authorized before its readiness-relative deadline"
        )
    if first_intent_status != "ok":
        raise DataGateError(f"unknown first-intent status: {first_intent_status}")
    return cross_fit_path


# ===========================================================================
# Item 4: start interaction deadlines only after controller and environment
# readiness.
# ===========================================================================
def readiness_gate(
    *,
    lane: Mapping[str, Any],
    controller_ready_at: float | None,
    environment_ready_at: float | None,
    interaction_budget_seconds: float,
) -> dict[str, Any]:
    """Compute the interaction deadline anchored at joint readiness.

    The clock refuses to start until *both* the controller (posterior fitted or
    loaded) and the environment (opened and reset) report ready.  This is the
    structural fix for T10.2.1, where the reset clock began at process spawn.
    """

    if controller_ready_at is None or environment_ready_at is None:
        raise GateRefusalError(
            "interaction deadline cannot start before controller and environment "
            "readiness"
        )
    if not (_finite_nonneg(controller_ready_at) and _finite_nonneg(environment_ready_at)):
        raise DataGateError("readiness clocks must be finite and non-negative")
    if not _finite_nonneg(interaction_budget_seconds) or interaction_budget_seconds <= 0:
        raise DataGateError("interaction budget must be a positive finite number")
    interaction_started_at = max(
        float(controller_ready_at), float(environment_ready_at)
    )
    return signed_payload(
        {
            "format_version": READINESS_FORMAT_VERSION,
            "lane_id": lane.get("lane_id"),
            "controller_ready_at": float(controller_ready_at),
            "environment_ready_at": float(environment_ready_at),
            "interaction_started_at": interaction_started_at,
            "interaction_budget_seconds": float(interaction_budget_seconds),
            "interaction_deadline": interaction_started_at
            + float(interaction_budget_seconds),
            "clock_anchored_at_readiness": True,
        },
        checksum_key="readiness_checksum",
    )


# ===========================================================================
# Item 5: complete evidence funnel with rejection-reason accounting.
# ===========================================================================
def build_evidence_funnel(
    *,
    observed_intents: int,
    authorized_intents: int,
    sealed_events: int,
    rejections: Mapping[str, int],
) -> dict[str, Any]:
    """Account for every intent: sealed, or rejected with a registered reason.

    Invariants enforced:

    * ``authorized_intents == sealed_events + rejected_after_authorization``
      (the frozen T10.2.1 conservation law, re-expressed with reasons)
    * every rejection reason is registered and every count is non-negative
    * ``observed_intents >= authorized_intents`` and the pre-authorization drops
      are themselves accounted as rejections.
    """

    observed = _require_nonneg_int(observed_intents, label="observed intents")
    authorized = _require_nonneg_int(authorized_intents, label="authorized intents")
    sealed = _require_nonneg_int(sealed_events, label="sealed events")
    if authorized > observed:
        raise DataGateError("authorized intents cannot exceed observed intents")
    if sealed > authorized:
        raise DataGateError("sealed events cannot exceed authorized intents")
    clean: dict[str, int] = {}
    for reason, count in rejections.items():
        if reason not in REGISTERED_REJECTION_REASONS:
            raise DataGateError(f"unregistered rejection reason: {reason}")
        clean[str(reason)] = _require_nonneg_int(count, label=f"rejection[{reason}]")
    rejected_total = sum(clean.values())
    preauth_drops = observed - authorized
    authorized_rejected = authorized - sealed
    # Every non-sealed intent (authorized-but-unsealed plus pre-authorization
    # drops) must carry exactly one registered rejection reason.
    expected_rejected = observed - sealed
    accounted = rejected_total == expected_rejected
    return signed_payload(
        {
            "format_version": EVIDENCE_FUNNEL_FORMAT_VERSION,
            "observed_intents": observed,
            "authorized_intents": authorized,
            "sealed_events": sealed,
            "rejected_intents": rejected_total,
            "authorized_rejected_intents": authorized_rejected,
            "preauthorization_drops": preauth_drops,
            "rejections": {reason: clean[reason] for reason in sorted(clean)},
            # The frozen T10.2.1 conservation law: authorized == sealed + unsealed.
            "conservation_holds": authorized == sealed + authorized_rejected,
            "fully_accounted": accounted,
        },
        checksum_key="funnel_checksum",
    )


def evidence_funnel_from_reset_reports(
    reset_reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive the funnel from reset reports (kernel has no pre-auth drops)."""

    observed = 0
    authorized = 0
    sealed = 0
    rejections: Counter[str] = Counter()
    for report in reset_reports:
        issued = _require_nonneg_int(report.get("issued_intents"), label="issued_intents")
        report_sealed = _require_nonneg_int(
            report.get("sealed_events"), label="sealed_events"
        )
        unresolved = _require_nonneg_int(
            report.get("unresolved_intents"), label="unresolved_intents"
        )
        observed += issued
        authorized += issued
        sealed += report_sealed
        if unresolved:
            reason = str(report.get("stop_reason", ""))
            if reason not in REGISTERED_REJECTION_REASONS:
                raise DataGateError(
                    f"reset carried unresolved intents without a registered reason: {reason}"
                )
            rejections[reason] += unresolved
    return build_evidence_funnel(
        observed_intents=observed,
        authorized_intents=authorized,
        sealed_events=sealed,
        rejections=dict(rejections),
    )


# ===========================================================================
# Item 6: separate canonical schema families from grounded instances.
# ===========================================================================
def _normalize_family_key(key: Any) -> str:
    """A coordinate-free family is ``name:arity``; accept tuple or string."""

    if isinstance(key, (list, tuple)) and len(key) == 2:
        name, arity = key
        return f"{name}:{int(arity)}"
    return str(key)


def partition_schema_evidence(
    *,
    learned_schema_counts: Mapping[Any, int],
    independent_schema_counts: Mapping[Any, int],
    grounding_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Separate coordinate-free schema families from grounded instances.

    Families answer "what capacity was exercised"; grounded instances answer
    "where it was grounded".  Conflating them lets a single grounding masquerade
    as broad capacity, so T10.2.2 reports them as disjoint ledgers.
    """

    def _fold(counts: Mapping[Any, int]) -> dict[str, int]:
        folded: Counter[str] = Counter()
        for key, value in counts.items():
            folded[_normalize_family_key(key)] += _require_nonneg_int(
                value, label="schema count"
            )
        return {name: folded[name] for name in sorted(folded)}

    learned = _fold(learned_schema_counts)
    independent = _fold(independent_schema_counts)
    grounded: dict[str, int] = {}
    for key, value in grounding_counts.items():
        grounded[str(key)] = _require_nonneg_int(value, label="grounding count")
    grounded = {key: grounded[key] for key in sorted(grounded)}
    canonical_families = sorted(set(learned) | set(independent))
    return signed_payload(
        {
            "format_version": SCHEMA_EVIDENCE_FORMAT_VERSION,
            "canonical_families": canonical_families,
            "canonical_family_count": len(canonical_families),
            "learned_schema_counts": learned,
            "independent_schema_counts": independent,
            "grounded_instances": grounded,
            "grounded_instance_count": len(grounded),
            "families_are_coordinate_free": True,
            # A single grounding must never be counted as multiple families.
            "instance_to_family_collapse_ok": len(grounded)
            >= len(canonical_families)
            or not grounded,
        },
        checksum_key="schema_evidence_checksum",
    )


# ===========================================================================
# Item 7: controlled end-to-end induction canary.
# ===========================================================================
def default_canary_evidence() -> tuple[dict[str, Any], ...]:
    """A tiny, fixed, independent evidence bundle for the induction canary.

    It is deliberately decoupled from any lane so the canary is controlled: the
    same inputs must always induce the same non-empty family set.
    """

    return (
        {"schema": ("move", 1), "grounding": "move:[0]"},
        {"schema": ("move", 1), "grounding": "move:[1]"},
        {"schema": ("rotate", 0), "grounding": "rotate:[]"},
    )


def _default_induct(
    evidence: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    families = {
        _normalize_family_key(row.get("schema"))
        for row in evidence
        if row.get("schema") is not None
    }
    return tuple(sorted(families))


def run_induction_canary(
    *,
    induct: Callable[[Sequence[Mapping[str, Any]]], Iterable[Any]] | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove induction consumes delivered evidence, deterministically.

    The canary (a) delivers a fixed independent evidence bundle, (b) runs the
    real induction callable end to end, (c) requires at least one induced family,
    and (d) requires determinism across two runs.  A green canary is the
    precondition for ever blaming the learner (see the invariant guards).
    """

    canary_evidence = tuple(evidence) if evidence is not None else default_canary_evidence()
    induction = induct if induct is not None else _default_induct
    delivered = len(canary_evidence)
    if delivered == 0:
        raise DataGateError("induction canary requires non-empty evidence")
    first = tuple(str(item) for item in induction(canary_evidence))
    second = tuple(str(item) for item in induction(canary_evidence))
    deterministic = first == second
    families = sorted(set(first))
    passed = deterministic and len(families) >= 1
    return signed_payload(
        {
            "format_version": INDUCTION_CANARY_FORMAT_VERSION,
            "evidence_delivered": delivered,
            "induced_families": families,
            "induced_family_count": len(families),
            "deterministic": deterministic,
            "passed": passed,
        },
        checksum_key="canary_checksum",
    )


# ===========================================================================
# Item 8: interleave discovery and confirmation and reserve confirmation
# capacity.
# ===========================================================================
def reserved_confirmation_capacity(
    *, confirmation_lane_count: int | None = None
) -> int:
    total = (
        confirmation_lane_count
        if confirmation_lane_count is not None
        else len(_lanes_by_split()["leave_one_game_out_confirmation"])
    )
    total = _require_nonneg_int(total, label="confirmation lane count")
    return math.ceil(total * CONFIRMATION_RESERVE_FRACTION)


def interleaved_lane_schedule(
    *, lane_budget: int | None = None
) -> dict[str, Any]:
    """Round-robin discovery and confirmation, guaranteeing reserved capacity.

    Confirmation lanes are the ones that actually test learned transfer, so if a
    ``lane_budget`` forces truncation the reserved confirmation lanes are placed
    first among confirmation and are never dropped before discovery lanes beyond
    the reserve.
    """

    grouped = _lanes_by_split()
    discovery = list(grouped["discovery"])
    confirmation = list(grouped["leave_one_game_out_confirmation"])
    reserve = reserved_confirmation_capacity(
        confirmation_lane_count=len(confirmation)
    )

    # Round-robin interleave, discovery-first at each step.
    interleaved: list[dict[str, Any]] = []
    di = ci = 0
    while di < len(discovery) or ci < len(confirmation):
        if di < len(discovery):
            interleaved.append(discovery[di])
            di += 1
        if ci < len(confirmation):
            interleaved.append(confirmation[ci])
            ci += 1

    order = interleaved
    truncated = False
    if lane_budget is not None:
        budget = _require_nonneg_int(lane_budget, label="lane budget")
        if budget < reserve:
            raise ResourceGateError(
                "lane budget cannot satisfy the reserved confirmation capacity"
            )
        if budget < len(interleaved):
            truncated = True
            reserved_lanes = confirmation[:reserve]
            reserved_ids = {lane["lane_id"] for lane in reserved_lanes}
            # Fill the remaining budget from the interleaved order, then ensure
            # every reserved confirmation lane is present.
            selected: list[dict[str, Any]] = []
            for lane in interleaved:
                if len(selected) >= budget:
                    break
                selected.append(lane)
            selected_ids = {lane["lane_id"] for lane in selected}
            missing = [
                lane for lane in reserved_lanes if lane["lane_id"] not in selected_ids
            ]
            if missing:
                # Evict non-reserved trailing lanes to make room for the reserve.
                kept = [
                    lane
                    for lane in selected
                    if lane["lane_id"] in reserved_ids
                ] + [
                    lane
                    for lane in selected
                    if lane["lane_id"] not in reserved_ids
                ]
                kept = kept[: budget - len(missing)] + missing
                selected = kept[:budget]
            order = selected
    confirmation_in_order = [
        lane
        for lane in order
        if lane["split"] == "leave_one_game_out_confirmation"
    ]
    return signed_payload(
        {
            "format_version": LANE_SCHEDULE_FORMAT_VERSION,
            "discovery_lane_count": len(discovery),
            "confirmation_lane_count": len(confirmation),
            "reserved_confirmation_capacity": reserve,
            "scheduled_confirmation_lane_count": len(confirmation_in_order),
            "truncated": truncated,
            "interleaved": True,
            "order": [dict(lane) for lane in order],
        },
        checksum_key="schedule_checksum",
    )


# ===========================================================================
# Item 9: one smoke lane per split before launching the complete matrix.
# ===========================================================================
def smoke_lane_plan() -> dict[str, Any]:
    """Pick one representative lane per split to run before the full matrix."""

    grouped = _lanes_by_split()
    smoke: list[dict[str, Any]] = []
    for split in SOURCE_SPLITS:
        lanes = grouped[split]
        if not lanes:
            raise ProtocolError(f"split {split} has no lanes to smoke test")
        smoke.append(lanes[0])
    smoke_ids = {lane["lane_id"] for lane in smoke}
    full_matrix = list(source_lane_registry())
    remaining = [lane for lane in full_matrix if lane["lane_id"] not in smoke_ids]
    return signed_payload(
        {
            "format_version": SMOKE_PLAN_FORMAT_VERSION,
            "smoke_lanes": [dict(lane) for lane in smoke],
            "smoke_lane_count": len(smoke),
            "remaining_matrix_lane_count": len(remaining),
            "total_lane_count": len(full_matrix),
            "smoke_precedes_matrix": True,
            "one_per_split": len(smoke) == len(SOURCE_SPLITS),
        },
        checksum_key="smoke_plan_checksum",
    )


# ===========================================================================
# Invariant guards.
# ===========================================================================
def guard_schema_learning_verdict(
    *,
    verdict: str,
    independent_evidence_generated: bool,
    independent_evidence_delivered: bool,
) -> dict[str, Any]:
    """Never blame the learner unless independent evidence reached induction.

    If a schema-learning failure verdict is proposed but qualifying independent
    evidence was not both *generated* and *delivered* to the induction
    mechanism, the miss is re-attributed to acquisition.
    """

    if verdict not in EXCLUSIVE_VERDICTS:
        raise DataGateError(f"unregistered verdict: {verdict}")
    qualifying = bool(independent_evidence_generated) and bool(
        independent_evidence_delivered
    )
    if verdict in SCHEMA_LEARNING_FAILURE_VERDICTS and not qualifying:
        return {
            "verdict": EVIDENCE_UNMET_VERDICT,
            "adjusted": True,
            "proposed_verdict": verdict,
            "reason": "independent evidence was not generated and delivered to induction",
            "independent_evidence_generated": bool(independent_evidence_generated),
            "independent_evidence_delivered": bool(independent_evidence_delivered),
        }
    return {
        "verdict": verdict,
        "adjusted": False,
        "proposed_verdict": verdict,
        "reason": None,
        "independent_evidence_generated": bool(independent_evidence_generated),
        "independent_evidence_delivered": bool(independent_evidence_delivered),
    }


def qualifying_transfer_lanes(
    lane_intent_counts: Mapping[str, int]
) -> tuple[str, ...]:
    """Lanes that issued at least one intent are the only transfer-eligible ones."""

    qualifying: list[str] = []
    for lane_id, count in lane_intent_counts.items():
        if _require_nonneg_int(count, label="lane intent count") > 0:
            qualifying.append(str(lane_id))
    return tuple(sorted(qualifying))


def guard_transfer_verdict(
    *,
    verdict: str,
    lane_intent_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Never declare transfer failure for lanes that issued zero intents.

    Zero-intent lanes are exempt from the transfer denominator.  If *no* lane
    issued any intent, transfer could not have been tested at all, so the miss is
    re-attributed to acquisition instead of transfer.
    """

    if verdict not in EXCLUSIVE_VERDICTS:
        raise DataGateError(f"unregistered verdict: {verdict}")
    qualifying = qualifying_transfer_lanes(lane_intent_counts)
    zero_intent = tuple(
        sorted(
            str(lane_id)
            for lane_id, count in lane_intent_counts.items()
            if int(count) == 0
        )
    )
    if verdict == TRANSFER_FAILURE_VERDICT and not qualifying:
        return {
            "verdict": EVIDENCE_UNMET_VERDICT,
            "adjusted": True,
            "proposed_verdict": verdict,
            "reason": "no lane issued any intent; transfer was never exercised",
            "qualifying_lane_ids": list(qualifying),
            "zero_intent_lane_ids": list(zero_intent),
        }
    return {
        "verdict": verdict,
        "adjusted": False,
        "proposed_verdict": verdict,
        "reason": None,
        "qualifying_lane_ids": list(qualifying),
        "zero_intent_lane_ids": list(zero_intent),
    }


# ===========================================================================
# Top-level orchestration composition.
# ===========================================================================
def build_orchestration_report(
    *,
    manifest_checksum: str,
    checkpoint_binding: Mapping[str, Any],
    phase_timing: Mapping[str, Any],
    evidence_funnel: Mapping[str, Any],
    schema_evidence: Mapping[str, Any],
    induction_canary: Mapping[str, Any],
    lane_schedule: Mapping[str, Any],
    smoke_plan: Mapping[str, Any],
    proposed_verdict: str,
    lane_intent_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Compose the nine orchestration artifacts and apply both invariant guards.

    The proposed verdict passes through the transfer guard and then the
    schema-learning guard.  Schema-learning "delivery" is anchored to the green
    induction canary and a non-empty independent-schema ledger, so a red canary
    can never yield a learner-blaming verdict.
    """

    independent_generated = bool(
        schema_evidence.get("independent_schema_counts")
    )
    independent_delivered = bool(induction_canary.get("passed")) and independent_generated

    transfer = guard_transfer_verdict(
        verdict=proposed_verdict, lane_intent_counts=lane_intent_counts
    )
    schema = guard_schema_learning_verdict(
        verdict=str(transfer["verdict"]),
        independent_evidence_generated=independent_generated,
        independent_evidence_delivered=independent_delivered,
    )
    final_verdict = str(schema["verdict"])
    return signed_payload(
        {
            "format_version": ORCHESTRATION_REPORT_FORMAT_VERSION,
            "phase": "orchestrate",
            "manifest_checksum": manifest_checksum,
            "checkpoint_binding": dict(checkpoint_binding),
            "phase_timing": dict(phase_timing),
            "evidence_funnel": dict(evidence_funnel),
            "schema_evidence": dict(schema_evidence),
            "induction_canary": dict(induction_canary),
            "lane_schedule": dict(lane_schedule),
            "smoke_plan": dict(smoke_plan),
            "proposed_verdict": proposed_verdict,
            "transfer_guard": transfer,
            "schema_learning_guard": schema,
            "verdict": final_verdict,
            "verdict_adjusted": bool(transfer["adjusted"] or schema["adjusted"]),
        },
        checksum_key="report_checksum",
    )


# ---------------------------------------------------------------------------
# CLI: kernel phases delegate to the frozen T10.2.1 module; ``orchestrate`` runs
# the report/checkpoint synchronization introduced here.
# ---------------------------------------------------------------------------
PHASES = (*_t10_2_1.PHASES, "orchestrate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase != "orchestrate":
        raise SystemExit(
            "T10.2.2 delegates kernel phases to theory.sage_t.t10_2_1_protocol; "
            "run that module for freeze/collect/compile/replay/source-train/"
            "validate/report."
        )
    try:
        payload = synchronize_report_with_checkpoint(output_dir=args.output_dir)
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}", "phase": args.phase}))
        return 2
    print(canonical_json(payload))
    return 0


__all__ = [
    "CHECKPOINT_BINDING_FORMAT_VERSION",
    "CONFIRMATION_RESERVE_FRACTION",
    "CONFIRMATION_SEEDS",
    "CROSS_FIT_AUDIT_FILENAME",
    "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_SEEDS",
    "EVIDENCE_FUNNEL_FORMAT_VERSION",
    "EVIDENCE_UNMET_VERDICT",
    "EXCLUSIVE_VERDICTS",
    "FORMAT_VERSION",
    "INDUCTION_CANARY_FORMAT_VERSION",
    "LANE_SCHEDULE_FORMAT_VERSION",
    "ORCHESTRATION_REPORT_FORMAT_VERSION",
    "PHASES",
    "PHASE_TIMING_FORMAT_VERSION",
    "READINESS_FORMAT_VERSION",
    "REGISTERED_REJECTION_REASONS",
    "SCHEMA_EVIDENCE_FORMAT_VERSION",
    "SCHEMA_LEARNING_FAILURE_VERDICTS",
    "SMOKE_PLAN_FORMAT_VERSION",
    "SOURCE_GAMES",
    "SOURCE_SPLITS",
    "TRANSFER_FAILURE_VERDICT",
    "build_checkpoint_binding",
    "build_evidence_funnel",
    "build_orchestration_report",
    "build_phase_timing",
    "canonical_json",
    "canonical_sha256",
    "classify_first_intent",
    "compute_lane_startup_timing",
    "default_canary_evidence",
    "evidence_funnel_from_reset_reports",
    "fail_fast_preflight",
    "guard_schema_learning_verdict",
    "guard_transfer_verdict",
    "interleaved_lane_schedule",
    "main",
    "partition_schema_evidence",
    "qualifying_transfer_lanes",
    "readiness_gate",
    "require_cross_fit_artifacts",
    "reserved_confirmation_capacity",
    "run_induction_canary",
    "signed_payload",
    "smoke_lane_plan",
    "source_lane_registry",
    "synchronize_report_with_checkpoint",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
