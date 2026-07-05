"""Lightweight operator search for V4."""

from __future__ import annotations

from ..schemas import MindProposal, ObservationV4


class OperatorSearcher:
    """Small beam-like expansion over currently plausible operators."""

    def search(self, obs: ObservationV4, proposal: MindProposal, memory) -> list[str]:
        phase = memory.fast.current_phase
        if phase not in {"project_emergence", "closure_pressure", "compression"}:
            return proposal.operator_plan

        if not proposal.operator_plan:
            return []

        if phase == "project_emergence":
            depth = 3
        elif phase == "closure_pressure":
            depth = 5
        else:
            depth = 6

        ranked_ops = memory.game.laws.ranked_operators or list(memory.game.inducer.operators.values())
        aligned: list[str] = []
        for operator in ranked_ops:
            if operator.kind == "lethal":
                continue
            bonus = 0.0
            if proposal.project.kind == "reach_region" and operator.kind in {"move", "escape"}:
                bonus += 0.20
            if proposal.project.kind in {"probe_unique_object", "exhaust_class", "closure_probe"} and operator.kind == "click":
                bonus += 0.20
            if proposal.project.kind == "transform_then_probe" and operator.kind == "global_transform":
                bonus += 0.25
            if operator.confidence + bonus >= 0.45:
                aligned.append(operator.operator_id)
            if len(aligned) >= depth:
                break

        merged: list[str] = []
        for operator_id in proposal.operator_plan + aligned:
            if operator_id not in merged:
                merged.append(operator_id)
            if len(merged) >= depth:
                break
        return merged
