"""Closed-loop relational successor controls for SAGE.T10.3.12e.

The transferable payload retains only an abstract goal-end role and a
continuation rule.  Grounded paths, action arguments, and the reset-local
visited frontier are deliberately ephemeral.
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
)
from .executor_correspondence_v10_3_12d import ExecutorRegistry
from .factorial_invariants_v10_3_12b import assert_transfer_safe
from .progress_witness_v10 import GroundedAction

FORMAT_VERSION = "sage-t10.3.12e-closed-loop-successor-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.12e-closed-loop-registry-v1"
ARMS = (
    "anchored_goal_dynamic_successor",
    "frozen_grounded_cursor",
    "stateless_goal_and_successor",
    "goal_end_swap",
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
class ClosedLoopProgram:
    """Transfer-safe policy declaration without any grounded identity."""

    arm: str
    initiation: str
    goal_anchor: str
    successor_selection: str
    grounding: str
    state_memory: str
    termination: str
    safety_horizon: int
    source_kind: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unsupported T10.3.12e arm: {self.arm}")
        if not 1 <= int(self.safety_horizon) <= 16:
            raise ValueError("closed-loop horizon must be in [1, 16]")
        assert_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "arm": self.arm,
            "initiation": self.initiation,
            "goal_anchor": self.goal_anchor,
            "successor_selection": self.successor_selection,
            "grounding": self.grounding,
            "state_memory": self.state_memory,
            "termination": self.termination,
            "safety_horizon": int(self.safety_horizon),
            "source_kind": self.source_kind,
        }

    @property
    def program_hash(self) -> str:
        return sha256_payload(self.safe_payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ClosedLoopProgram":
        assert_transfer_safe(payload)
        return cls(
            arm=str(payload["arm"]),
            initiation=str(payload["initiation"]),
            goal_anchor=str(payload["goal_anchor"]),
            successor_selection=str(payload["successor_selection"]),
            grounding=str(payload["grounding"]),
            state_memory=str(payload["state_memory"]),
            termination=str(payload["termination"]),
            safety_horizon=int(payload["safety_horizon"]),
            source_kind=str(payload["source_kind"]),
        )


class ClosedLoopRegistry:
    """Fixed causal arms with zero imported target support."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self.parent_executor_registry_checksum = ""
        self.source_factor_registry_checksum = ""
        self._programs: dict[str, ClosedLoopProgram] = {}
        if payload is None:
            return
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("closed-loop registry format drifted")
        if int(payload.get("local_support_total", -1)) != 0:
            raise ValueError("closed-loop registry imported active support")
        self.parent_executor_registry_checksum = str(
            payload.get("parent_executor_registry_checksum", "")
        )
        self.source_factor_registry_checksum = str(
            payload.get("source_factor_registry_checksum", "")
        )
        for row in payload.get("programs", ()):
            program = ClosedLoopProgram.from_payload(row["program"])
            if row.get("program_hash") != program.program_hash:
                raise ValueError("closed-loop program hash drifted")
            if int(row.get("local_support", -1)) != 0:
                raise ValueError("closed-loop program contains local support")
            self.register(program)

    def register(self, program: ClosedLoopProgram) -> None:
        if program.arm in self._programs:
            raise ValueError(f"duplicate closed-loop program: {program.arm}")
        self._programs[program.arm] = program

    def program_for(self, arm: str) -> ClosedLoopProgram:
        try:
            return self._programs[arm]
        except KeyError as exc:
            raise KeyError(f"missing closed-loop program for {arm}") from exc

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
                "parent_executor_registry_checksum": self.parent_executor_registry_checksum,
                "source_factor_registry_checksum": self.source_factor_registry_checksum,
                "programs": rows,
                "local_support_total": 0,
                "promotion_count": 0,
            },
            "registry_checksum",
        )


def compile_closed_loop_registry(
    parent_executor_payload: Mapping[str, Any],
    source_factor_payload: Mapping[str, Any],
) -> ClosedLoopRegistry:
    """Compile only from the frozen abstract source factor and d executor."""

    parent = ExecutorRegistry(parent_executor_payload)
    stable = parent.program_for("stable_source_cursor")
    expected_executor = {
        "initiation": "fresh_reset_path_relation",
        "orientation": "source_salient_end",
        "continuation": "option_local_cursor",
        "reacquisition": "exact_current_legal_action",
        "termination": "level_progress_ambiguity_or_plan_exhaustion",
        "safety_horizon": 16,
    }
    observed_executor = {
        "initiation": stable.initiation,
        "orientation": stable.orientation,
        "continuation": stable.continuation,
        "reacquisition": stable.reacquisition,
        "termination": stable.termination,
        "safety_horizon": stable.safety_horizon,
    }
    if observed_executor != expected_executor:
        raise ValueError("T10.3.12d stable executor declaration drifted")

    source = CrossGameFactorRegistry(source_factor_payload)
    path_factor = source.program_for("factorized_source", "path_context")
    expected_factor = {
        "operator": "parameterized_apply",
        "role_binding": "salient_end_prior_with_causal_verification",
        "transition": "successor_toward_goal_end",
        "termination": "stop_on_progress_or_ambiguity",
        "safety_horizon": 16,
    }
    observed_factor = {
        "operator": path_factor.operator,
        "role_binding": path_factor.role_binding,
        "transition": path_factor.transition,
        "termination": path_factor.termination,
        "safety_horizon": path_factor.safety_horizon,
    }
    if observed_factor != expected_factor:
        raise ValueError("frozen source path factor drifted")
    if (
        parent_executor_payload.get("parent_factor_registry_checksum")
        != source_factor_payload.get("registry_checksum")
    ):
        raise ValueError("T10.3.12d executor is detached from the source factor registry")

    declarations = {
        "anchored_goal_dynamic_successor": {
            "goal_anchor": "source_salient_endpoint_role",
            "successor_selection": "first_unvisited_current_relational_successor",
            "state_memory": "reset_local_visited_relational_frontier",
            "source_kind": "closed_loop_source_factor_executor",
        },
        "frozen_grounded_cursor": {
            "goal_anchor": "source_salient_endpoint_role",
            "successor_selection": "fixed_initial_grounded_cursor",
            "state_memory": "reset_local_frozen_grounded_path_control",
            "source_kind": "t10_3_12d_frozen_cursor_control",
        },
        "stateless_goal_and_successor": {
            "goal_anchor": "reestimated_salient_endpoint_role",
            "successor_selection": "first_current_relational_successor",
            "state_memory": "none",
            "source_kind": "stateless_replanning_control",
        },
        "goal_end_swap": {
            "goal_anchor": "anti_salient_endpoint_role",
            "successor_selection": "first_unvisited_reverse_relational_successor",
            "state_memory": "reset_local_visited_relational_frontier",
            "source_kind": "goal_role_causal_control",
        },
    }
    registry = ClosedLoopRegistry()
    registry.parent_executor_registry_checksum = str(
        parent_executor_payload.get("registry_checksum", "")
    )
    registry.source_factor_registry_checksum = str(
        source_factor_payload.get("registry_checksum", "")
    )
    for arm in ARMS:
        row = declarations[arm]
        registry.register(
            ClosedLoopProgram(
                arm=arm,
                initiation="fresh_reset_path_relation",
                goal_anchor=row["goal_anchor"],
                successor_selection=row["successor_selection"],
                grounding="exact_current_legal_action",
                state_memory=row["state_memory"],
                termination="level_progress_ambiguity_or_relational_frontier_exhaustion",
                safety_horizon=16,
                source_kind=row["source_kind"],
            )
        )
    return registry


@dataclass(frozen=True)
class ClosedLoopDecision:
    candidate: ActionCandidate | None
    reason: str
    program_hash: str
    current_path_length: int
    frontier_size: int

    @property
    def abstained(self) -> bool:
        return self.candidate is None


def _same(candidate: ActionCandidate, wanted: GroundedAction) -> bool:
    return candidate.action_name == wanted.action_name and dict(candidate.action_data) == wanted.data


PathBuilder = Callable[
    [AbstractState | None, Sequence[ActionCandidate]],
    tuple[str, tuple[GroundedAction, ...]],
]


class ClosedLoopSuccessorController:
    """Reset-local closed-loop executor with no serialised grounded state."""

    def __init__(
        self,
        *,
        arm: str,
        registry: ClosedLoopRegistry,
        path_builder: PathBuilder = recognize_context,
    ) -> None:
        self.program = registry.program_for(arm)
        self.path_builder = path_builder
        self._initialized = False
        self._recognized_context = ""
        self._anchor_builds = 0
        self._relation_evaluations = 0
        self._dynamic_regrounds = 0
        self._frontier_advances = 0
        self._repeat_proposals_rejected = 0
        self._exact_groundings = 0
        self._grounding_misses = 0
        self._abstention_reason = ""
        self._initial_path: tuple[GroundedAction, ...] = ()
        self._frozen_cursor = 0
        self._visited_action_keys: set[str] = set()

    def _abstain(self, reason: str, path_length: int = 0) -> ClosedLoopDecision:
        self._abstention_reason = reason
        return ClosedLoopDecision(
            None,
            reason,
            self.program.program_hash,
            path_length,
            len(self._visited_action_keys),
        )

    def _observe_path(
        self,
        state: AbstractState | None,
        legal: Sequence[ActionCandidate],
    ) -> tuple[str, tuple[GroundedAction, ...]]:
        context, path = self.path_builder(state, legal)
        self._relation_evaluations += 1
        return context, tuple(path)

    def _ground(
        self,
        legal: Sequence[ActionCandidate],
        wanted: GroundedAction,
        *,
        reason: str,
        path_length: int,
    ) -> ClosedLoopDecision:
        matches = [candidate for candidate in legal if _same(candidate, wanted)]
        if len(matches) != 1:
            self._grounding_misses += 1
            return self._abstain("current_relational_successor_grounding_miss", path_length)
        self._exact_groundings += 1
        return ClosedLoopDecision(
            matches[0],
            reason,
            self.program.program_hash,
            path_length,
            len(self._visited_action_keys),
        )

    def choose(
        self,
        *,
        state: AbstractState | None,
        candidates: Sequence[Any],
        shape: tuple[int, int],
        step_index: int,
    ) -> ClosedLoopDecision:
        del shape
        legal = normalized_action_candidates(candidates)
        if step_index >= self.program.safety_horizon:
            return self._abstain("closed_loop_horizon_exhausted")

        if self.program.arm == "frozen_grounded_cursor":
            if not self._initialized:
                context, path = self._observe_path(state, legal)
                self._initialized = True
                self._recognized_context = context
                if context != "path_context" or not path:
                    return self._abstain("non_path_context")
                self._anchor_builds = 1
                self._initial_path = path
            if self._frozen_cursor >= len(self._initial_path):
                return self._abstain("frozen_plan_exhausted", len(self._initial_path))
            wanted = self._initial_path[self._frozen_cursor]
            decision = self._ground(
                legal,
                wanted,
                reason="frozen_exact_waypoint",
                path_length=len(self._initial_path),
            )
            if not decision.abstained:
                self._frozen_cursor += 1
            return decision

        context, path = self._observe_path(state, legal)
        self._dynamic_regrounds += 1
        if not self._initialized:
            self._initialized = True
            self._recognized_context = context
            if context == "path_context" and path:
                self._anchor_builds = 1
        if self._recognized_context != "path_context":
            return self._abstain("non_path_context")
        if context != "path_context" or not path:
            self._grounding_misses += 1
            return self._abstain("current_path_relation_unavailable")

        oriented = tuple(reversed(path)) if self.program.arm == "goal_end_swap" else path
        if self.program.arm == "stateless_goal_and_successor":
            wanted = oriented[0]
            return self._ground(
                legal,
                wanted,
                reason="stateless_current_first_successor",
                path_length=len(oriented),
            )

        wanted = None
        for proposal in oriented:
            if proposal.key in self._visited_action_keys:
                self._repeat_proposals_rejected += 1
                continue
            wanted = proposal
            break
        if wanted is None:
            return self._abstain("relational_frontier_exhausted", len(oriented))
        decision = self._ground(
            legal,
            wanted,
            reason=(
                "anchored_dynamic_relational_successor"
                if self.program.arm == "anchored_goal_dynamic_successor"
                else "goal_end_swap_dynamic_successor"
            ),
            path_length=len(oriented),
        )
        if not decision.abstained:
            self._visited_action_keys.add(wanted.key)
            self._frontier_advances += 1
        return decision

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.program.arm,
            "program_hash": self.program.program_hash,
            "recognized_context": self._recognized_context,
            "anchor_builds": self._anchor_builds,
            "relation_evaluations": self._relation_evaluations,
            "dynamic_regrounds": self._dynamic_regrounds,
            "frontier_advances": self._frontier_advances,
            "repeat_proposals_rejected": self._repeat_proposals_rejected,
            "exact_groundings": self._exact_groundings,
            "grounding_misses": self._grounding_misses,
            "frontier_size": len(self._visited_action_keys),
            "frozen_cursor": self._frozen_cursor,
            "initial_path_length": len(self._initial_path),
            "abstention_reason": self._abstention_reason,
            "path_plan_persisted": False,
            "visited_action_keys_persisted": False,
            "grounded_arguments_persisted": False,
            "cross_reset_memory": False,
        }


__all__ = [
    "ARMS",
    "FORMAT_VERSION",
    "REGISTRY_FORMAT_VERSION",
    "ClosedLoopDecision",
    "ClosedLoopProgram",
    "ClosedLoopRegistry",
    "ClosedLoopSuccessorController",
    "canonical_json",
    "compile_closed_loop_registry",
    "sha256_payload",
    "signed",
]
