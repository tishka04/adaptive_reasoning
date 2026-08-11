"""Ownership boundary for the one-executor, one-posterior SAGE.T runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from .contracts import ActionProgram, CausalProgram, CausalState, TransitionEvidence
from .decision import CausalDecision, CausalDecisionEngine
from .diagnostics import CausalDiagnosticsWriter
from .executor import CausalExecutor
from .memory import CausalMemoryStore
from .posterior import CausalPosterior, PosteriorUpdate


class CausalRuntime:
    def __init__(
        self,
        *,
        executor: CausalExecutor | None = None,
        posterior: CausalPosterior | None = None,
        decision_engine: CausalDecisionEngine | None = None,
        memory_path: str | Path | None = None,
        diagnostics: CausalDiagnosticsWriter | None = None,
    ) -> None:
        self.executor = executor or CausalExecutor()
        self.posterior = posterior or CausalPosterior(executor=self.executor)
        self.decision_engine = decision_engine or CausalDecisionEngine(
            executor=self.executor
        )
        if self.posterior.executor is not self.executor:
            raise ValueError("causal posterior must use the runtime's unique executor")
        if self.decision_engine.executor is not self.executor:
            raise ValueError("causal decision engine must use the runtime's unique executor")
        self.memory = CausalMemoryStore(memory_path) if memory_path is not None else None
        self.diagnostics = diagnostics

    def seed(self, programs: Sequence[CausalProgram]) -> None:
        self.posterior.seed(programs)

    def observe(self, evidence: TransitionEvidence) -> PosteriorUpdate:
        update = self.posterior.update(evidence)
        if self.memory is not None:
            self.memory.consolidate(
                update=update,
                posterior=self.posterior,
                evidence=evidence,
            )
        if self.diagnostics is not None:
            self.diagnostics.write(
                "posterior",
                {
                    "evidence_id": evidence.evidence_id,
                    "entropy_before": update.entropy_before,
                    "entropy_after": update.entropy_after,
                    "effective_sample_size": update.effective_sample_size,
                    "posterior": self.posterior.snapshot(maximum_particles=None),
                },
            )
            for program_hash in update.repair_children:
                self.diagnostics.write(
                    "repair",
                    {"evidence_id": evidence.evidence_id, "program_hash": program_hash},
                )
        return update

    def reload_memory(self) -> int:
        return 0 if self.memory is None else self.memory.reload(self.posterior)

    def decide(
        self,
        state: CausalState,
        candidates: Sequence[ActionProgram],
        *,
        danger_veto: Callable[[ActionProgram], bool] | None = None,
    ) -> CausalDecision:
        return self.decision_engine.decide(
            self.posterior,
            state,
            candidates,
            danger_veto=danger_veto,
        )


__all__ = ["CausalRuntime"]
