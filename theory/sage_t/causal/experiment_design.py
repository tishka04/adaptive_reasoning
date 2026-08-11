"""Bounded candidate generation and preregistered causal-probe matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import (
    ActionProgram,
    CausalState,
    GroundedAction,
    InterventionBranch,
    InterventionBundle,
)
from .runtime import CausalRuntime


@dataclass(frozen=True)
class CandidateSet:
    programs: tuple[ActionProgram, ...]
    truncated: bool = False


class CausalCandidateGenerator:
    def __init__(self, *, maximum_candidates: int = 64) -> None:
        self.maximum_candidates = max(1, min(64, int(maximum_candidates)))

    def generate(
        self,
        *,
        legal_actions: Sequence[GroundedAction],
        exact_routes: Sequence[ActionProgram] = (),
        progressive_routes: Sequence[ActionProgram] = (),
        frontier_programs: Sequence[ActionProgram] = (),
        multiform_programs: Sequence[ActionProgram] = (),
        include_causal_probes: bool = True,
    ) -> CandidateSet:
        ordered = []
        ordered.extend(_with_source(exact_routes, "exact_route"))
        ordered.extend(_with_source(progressive_routes, "progressive_route"))
        if include_causal_probes:
            ordered.extend(
                ActionProgram((action,), source="causal_probe")
                for action in legal_actions
            )
        ordered.extend(_with_source(frontier_programs, "frontier"))
        ordered.extend(_with_source(multiform_programs, "multiform"))
        ordered.extend(
            ActionProgram((action,), source="generic") for action in legal_actions
        )
        unique = []
        seen: set[tuple[str, str]] = set()
        for program in ordered:
            key = (program.key, program.source)
            if key in seen:
                continue
            seen.add(key)
            unique.append(program)
        return CandidateSet(
            programs=tuple(unique[: self.maximum_candidates]),
            truncated=len(unique) > self.maximum_candidates,
        )

    def prediction_matrix(
        self,
        *,
        runtime: CausalRuntime,
        state: CausalState,
        candidates: Sequence[ActionProgram],
    ) -> Mapping[str, Mapping[str, tuple[object, ...]]]:
        decision = runtime.decide(state, candidates)
        return {
            assessment.action_program.key: {
                prediction.program_hash: tuple(
                    packet.structured_signature
                    for packet in prediction.trace.predictions
                )
                for prediction in assessment.predictions
            }
            for assessment in decision.assessments
        }

    def intervention_bundle(
        self,
        *,
        prefix: ActionProgram,
        prefix_hash: str,
        branch_actions: Sequence[GroundedAction],
        prediction_matrix: Mapping[str, Mapping[str, tuple[object, ...]]],
    ) -> InterventionBundle:
        branches = []
        for action in branch_actions:
            candidate = ActionProgram((action,), source="causal_probe")
            predictions = prediction_matrix.get(candidate.key)
            if not predictions:
                raise ValueError(
                    f"missing preregistered predictions for branch {candidate.key}"
                )
            branches.append(
                InterventionBranch(
                    action=action,
                    predicted_signatures=predictions,
                )
            )
        return InterventionBundle(
            prefix=prefix,
            prefix_hash=prefix_hash,
            branches=tuple(branches),
        )


def _with_source(
    programs: Sequence[ActionProgram], source: str
) -> tuple[ActionProgram, ...]:
    return tuple(ActionProgram(program.actions, source=source) for program in programs)


__all__ = ["CandidateSet", "CausalCandidateGenerator"]
