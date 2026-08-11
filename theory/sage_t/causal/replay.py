"""Exact-prefix intervention bundles with preregistered predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import InterventionBundle, TransitionEvidence
from .posterior import PosteriorUpdate
from .runtime import CausalRuntime


class PrefixReplayEnvironment(Protocol):
    def reset_and_replay(self, prefix: object) -> Any:
        ...

    def state_hash(self, frame: Any) -> str:
        ...

    def legal_action_names(self, frame: Any) -> Sequence[str]:
        ...

    def execute(self, frame: Any, action: object) -> Any:
        ...


EvidenceBuilder = Callable[[Any, object, Any, int], TransitionEvidence]


@dataclass(frozen=True)
class BranchExecution:
    action_name: str
    prefix_hash: str
    evidence_id: str
    entropy_before: float
    entropy_after: float


@dataclass(frozen=True)
class InterventionBundleResult:
    status: str
    predictions_registered_before_execution: bool
    branches: tuple[BranchExecution, ...] = ()
    reason: str = ""

    @property
    def entropy_reduction(self) -> float:
        if not self.branches:
            return 0.0
        return self.branches[0].entropy_before - self.branches[-1].entropy_after


class InterventionBundleRunner:
    def __init__(self, *, runtime: CausalRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        bundle: InterventionBundle,
        *,
        environment: PrefixReplayEnvironment,
        evidence_builder: EvidenceBuilder,
    ) -> InterventionBundleResult:
        preregistered = all(branch.predicted_signatures for branch in bundle.branches)
        if not preregistered:
            return InterventionBundleResult(
                status="BUNDLE_REJECTED",
                predictions_registered_before_execution=False,
                reason="missing_preexecution_prediction_matrix",
            )
        executions = []
        for index, branch in enumerate(bundle.branches):
            before = environment.reset_and_replay(bundle.prefix)
            observed_hash = environment.state_hash(before)
            if observed_hash != bundle.prefix_hash:
                return InterventionBundleResult(
                    status="PREFIX_HASH_MISMATCH",
                    predictions_registered_before_execution=True,
                    branches=tuple(executions),
                    reason=f"branch_{index}_prefix_hash_mismatch",
                )
            if branch.action.action_name not in set(environment.legal_action_names(before)):
                return InterventionBundleResult(
                    status="BRANCH_ACTION_UNAVAILABLE",
                    predictions_registered_before_execution=True,
                    branches=tuple(executions),
                    reason=f"branch_{index}_action_unavailable",
                )
            entropy_before = self.runtime.posterior.entropy
            after = environment.execute(before, branch.action)
            evidence = evidence_builder(before, branch.action, after, index)
            update: PosteriorUpdate = self.runtime.observe(evidence)
            executions.append(
                BranchExecution(
                    action_name=branch.action.action_name,
                    prefix_hash=observed_hash,
                    evidence_id=evidence.evidence_id,
                    entropy_before=entropy_before,
                    entropy_after=update.entropy_after,
                )
            )
        return InterventionBundleResult(
            status="BUNDLE_COMPLETE",
            predictions_registered_before_execution=True,
            branches=tuple(executions),
        )


__all__ = [
    "BranchExecution", "InterventionBundleResult", "InterventionBundleRunner",
    "PrefixReplayEnvironment",
]
