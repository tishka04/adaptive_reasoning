"""Bounded, typed local repair operators for causal particles."""

from __future__ import annotations

from dataclasses import replace

from .comparison import ParticleComparison
from .contracts import CausalProgram, TransitionEvidence


class CausalProgramRepairer:
    """Generate local children; never regenerate an unconstrained program."""

    def __init__(self, *, maximum_children: int = 8) -> None:
        self.maximum_children = max(0, min(8, int(maximum_children)))

    def propose(
        self,
        program: CausalProgram,
        evidence: TransitionEvidence,
        comparison: ParticleComparison,
    ) -> tuple[CausalProgram, ...]:
        if self.maximum_children == 0 or not evidence.observed_delta.variable_changes:
            return ()
        children = []
        mechanisms = {item.output_variable: item for item in program.mechanisms}
        for variable_id, observed in evidence.observed_delta.variable_changes.items():
            parent = mechanisms.get(variable_id)
            if parent is None:
                continue
            repaired = replace(
                parent,
                mechanism_id=(
                    f"repair_{program.canonical_hash[:16]}_{len(children)}"
                ),
                operator_type="set",
                parameters={
                    "value": observed.mode,
                    "action_name": evidence.action.action_name,
                },
                neural_module_id=None,
                symbolic_fallback=None,
            )
            child_mechanisms = tuple(
                repaired if item.output_variable == variable_id else item
                for item in program.mechanisms
            )
            children.append(
                replace(
                    program,
                    program_id=(
                        f"repair_{program.canonical_hash[:16]}_{len(children)}"
                    ),
                    mechanisms=child_mechanisms,
                    description_length=float(program.description_length) + 1.0,
                    provenance=program.provenance
                    + (f"repair:{program.canonical_hash}:{evidence.evidence_id}",),
                )
            )
            if len(children) >= self.maximum_children:
                break
        return tuple(children)


__all__ = ["CausalProgramRepairer"]
