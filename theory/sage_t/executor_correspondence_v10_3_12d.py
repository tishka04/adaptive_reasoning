"""Option-local executor correspondence controls for SAGE.T10.3.12d.

T10.3.12d is diagnostic.  It reuses the frozen T10.3.12c source program but
changes only how a freshly grounded path is continued inside one reset.  Every
grounded waypoint remains ephemeral and is reacquired exactly from the current
legal action set before execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .contracts import AbstractState, ActionCandidate, normalized_action_candidates
from .cross_game_transfer_v10_3_12c import (
    CrossGameFactorRegistry,
    recognize_context,
    select_grounding,
)
from .factorial_invariants_v10_3_12b import assert_transfer_safe
from .progress_witness_v10 import GroundedAction

FORMAT_VERSION = "sage-t10.3.12d-executor-correspondence-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.12d-executor-registry-v1"
ARMS = (
    "stable_source_cursor",
    "stateless_source_replan",
    "stable_reverse_orientation",
    "stable_cursor_hold",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


@dataclass(frozen=True)
class ExecutorProgram:
    """Transfer-safe execution policy; never contains a grounded path."""

    arm: str
    initiation: str
    orientation: str
    continuation: str
    reacquisition: str
    termination: str
    safety_horizon: int
    source_kind: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unsupported T10.3.12d arm: {self.arm}")
        if not 1 <= int(self.safety_horizon) <= 16:
            raise ValueError("executor horizon must be in [1, 16]")
        assert_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "arm": self.arm,
            "initiation": self.initiation,
            "orientation": self.orientation,
            "continuation": self.continuation,
            "reacquisition": self.reacquisition,
            "termination": self.termination,
            "safety_horizon": int(self.safety_horizon),
            "source_kind": self.source_kind,
        }

    @property
    def program_hash(self) -> str:
        return sha256_payload(self.safe_payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutorProgram":
        assert_transfer_safe(payload)
        return cls(
            arm=str(payload["arm"]),
            initiation=str(payload["initiation"]),
            orientation=str(payload["orientation"]),
            continuation=str(payload["continuation"]),
            reacquisition=str(payload["reacquisition"]),
            termination=str(payload["termination"]),
            safety_horizon=int(payload["safety_horizon"]),
            source_kind=str(payload["source_kind"]),
        )


class ExecutorRegistry:
    """Four fixed executor policies with zero imported target support."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self.parent_factor_registry_checksum = ""
        self._programs: dict[str, ExecutorProgram] = {}
        if payload is None:
            return
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("executor registry format drifted")
        if int(payload.get("local_support_total", -1)) != 0:
            raise ValueError("executor registry imported active support")
        self.parent_factor_registry_checksum = str(
            payload.get("parent_factor_registry_checksum", "")
        )
        for row in payload.get("programs", ()):
            program = ExecutorProgram.from_payload(row["program"])
            if row.get("program_hash") != program.program_hash:
                raise ValueError("executor program hash drifted")
            if int(row.get("local_support", -1)) != 0:
                raise ValueError("executor program contains local support")
            self.register(program)

    def register(self, program: ExecutorProgram) -> None:
        if program.arm in self._programs:
            raise ValueError(f"duplicate executor program: {program.arm}")
        self._programs[program.arm] = program

    def program_for(self, arm: str) -> ExecutorProgram:
        try:
            return self._programs[arm]
        except KeyError as exc:
            raise KeyError(f"missing executor program for {arm}") from exc

    def snapshot(self) -> dict[str, Any]:
        rows = []
        for arm in sorted(self._programs):
            program = self._programs[arm]
            rows.append(
                {
                    "arm": arm,
                    "program_hash": program.program_hash,
                    "program": program.safe_payload,
                    "local_support": 0,
                    "promoted": False,
                }
            )
        return signed(
            {
                "format_version": REGISTRY_FORMAT_VERSION,
                "parent_factor_registry_checksum": self.parent_factor_registry_checksum,
                "programs": rows,
                "local_support_total": 0,
                "promotion_count": 0,
            },
            "registry_checksum",
        )


def compile_executor_registry(
    parent_payload: Mapping[str, Any],
) -> ExecutorRegistry:
    """Compile executor controls from the exact frozen source path factor."""

    parent = CrossGameFactorRegistry(parent_payload)
    source = parent.program_for("factorized_source", "path_context")
    expected = {
        "operator": "parameterized_apply",
        "role_binding": "salient_end_prior_with_causal_verification",
        "transition": "successor_toward_goal_end",
        "termination": "stop_on_progress_or_ambiguity",
        "safety_horizon": 16,
    }
    observed = {
        "operator": source.operator,
        "role_binding": source.role_binding,
        "transition": source.transition,
        "termination": source.termination,
        "safety_horizon": source.safety_horizon,
    }
    if observed != expected:
        raise ValueError("T10.3.12c source path factor drifted")
    registry = ExecutorRegistry()
    registry.parent_factor_registry_checksum = str(
        parent_payload.get("registry_checksum", "")
    )
    policies = {
        "stable_source_cursor": {
            "orientation": "source_salient_end",
            "continuation": "option_local_cursor",
        },
        "stateless_source_replan": {
            "orientation": "source_salient_end",
            "continuation": "recompute_and_take_first",
        },
        "stable_reverse_orientation": {
            "orientation": "reverse_source_end",
            "continuation": "option_local_cursor",
        },
        "stable_cursor_hold": {
            "orientation": "source_salient_end",
            "continuation": "hold_initial_waypoint",
        },
    }
    for arm in ARMS:
        registry.register(
            ExecutorProgram(
                arm=arm,
                initiation="fresh_reset_path_relation",
                orientation=policies[arm]["orientation"],
                continuation=policies[arm]["continuation"],
                reacquisition="exact_current_legal_action",
                termination="level_progress_ambiguity_or_plan_exhaustion",
                safety_horizon=16,
                source_kind=(
                    "faithful_t10_3_7_executor_correspondence"
                    if arm == "stable_source_cursor"
                    else "preregistered_executor_control"
                ),
            )
        )
    return registry


@dataclass(frozen=True)
class ExecutorDecision:
    candidate: ActionCandidate | None
    reason: str
    program_hash: str
    plan_length: int
    cursor: int

    @property
    def abstained(self) -> bool:
        return self.candidate is None


def _grounded(candidate: ActionCandidate) -> GroundedAction:
    return GroundedAction(
        candidate.action_name,
        tuple(dict(candidate.action_data).items()),
    )


def _same(candidate: ActionCandidate, wanted: GroundedAction) -> bool:
    return candidate.action_name == wanted.action_name and dict(candidate.action_data) == wanted.data


class PathExecutorController:
    """One reset-local path executor with no cross-reset or serialised plan."""

    def __init__(
        self,
        *,
        arm: str,
        registry: ExecutorRegistry,
        parent_registry: CrossGameFactorRegistry,
        plan_builder: Callable[
            [AbstractState | None, Sequence[ActionCandidate]],
            tuple[str, tuple[GroundedAction, ...]],
        ] = recognize_context,
        stateless_grounder: Callable[..., Any] = select_grounding,
    ) -> None:
        self.program = registry.program_for(arm)
        self.parent_registry = parent_registry
        self.plan_builder = plan_builder
        self.stateless_grounder = stateless_grounder
        self._plan: tuple[GroundedAction, ...] = ()
        self._initialized = False
        self._cursor = 0
        self._plan_builds = 0
        self._replans = 0
        self._reacquisitions = 0
        self._grounding_misses = 0
        self._abstention_reason = ""
        self._recognized_context = ""

    def _abstain(self, reason: str) -> ExecutorDecision:
        self._abstention_reason = reason
        return ExecutorDecision(
            None,
            reason,
            self.program.program_hash,
            len(self._plan),
            self._cursor,
        )

    def _initialize(
        self,
        state: AbstractState | None,
        candidates: Sequence[ActionCandidate],
    ) -> ExecutorDecision | None:
        context, path = self.plan_builder(state, candidates)
        self._initialized = True
        self._recognized_context = context
        if context != "path_context" or not path:
            return self._abstain("non_path_context")
        self._plan_builds += 1
        self._plan = tuple(path)
        if self.program.orientation == "reverse_source_end":
            self._plan = tuple(reversed(self._plan))
        return None

    def choose(
        self,
        *,
        state: AbstractState | None,
        candidates: Sequence[Any],
        shape: tuple[int, int],
        step_index: int,
    ) -> ExecutorDecision:
        legal = normalized_action_candidates(candidates)
        if not self._initialized:
            initialization = self._initialize(state, legal)
            if initialization is not None:
                return initialization

        if self.program.continuation == "recompute_and_take_first":
            self._replans += 1
            grounding = self.stateless_grounder(
                self.parent_registry,
                arm="factorized_source",
                candidates=legal,
                shape=shape,
                step_index=step_index,
                state=state,
            )
            if grounding.abstained or grounding.context != "path_context":
                self._grounding_misses += 1
                return self._abstain("stateless_replan_miss")
            return ExecutorDecision(
                grounding.candidate,
                "stateless_recompute_first",
                self.program.program_hash,
                0,
                step_index,
            )

        if self.program.continuation == "hold_initial_waypoint":
            plan_index = 0
        else:
            plan_index = self._cursor
        if plan_index >= len(self._plan):
            return self._abstain("stable_plan_exhausted")
        wanted = self._plan[plan_index]
        matches = [candidate for candidate in legal if _same(candidate, wanted)]
        if len(matches) != 1:
            self._grounding_misses += 1
            return self._abstain("stable_waypoint_reacquisition_miss")
        selected = matches[0]
        self._reacquisitions += 1
        if self.program.continuation == "option_local_cursor":
            self._cursor += 1
        return ExecutorDecision(
            selected,
            (
                "stable_exact_waypoint"
                if self.program.continuation == "option_local_cursor"
                else "cursor_hold_initial_waypoint"
            ),
            self.program.program_hash,
            len(self._plan),
            plan_index,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.program.arm,
            "program_hash": self.program.program_hash,
            "recognized_context": self._recognized_context,
            "plan_builds": self._plan_builds,
            "replans": self._replans,
            "plan_length": len(self._plan),
            "cursor": self._cursor,
            "reacquisitions": self._reacquisitions,
            "grounding_misses": self._grounding_misses,
            "abstention_reason": self._abstention_reason,
            "path_plan_persisted": False,
            "grounded_arguments_persisted": False,
            "cross_reset_memory": False,
        }


__all__ = [
    "ARMS",
    "FORMAT_VERSION",
    "REGISTRY_FORMAT_VERSION",
    "ExecutorDecision",
    "ExecutorProgram",
    "ExecutorRegistry",
    "PathExecutorController",
    "canonical_json",
    "compile_executor_registry",
    "sha256_payload",
    "signed",
]
