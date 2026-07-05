"""Dissent controller for V4."""

from __future__ import annotations

import random

from ..schemas import ActionIntent, DissentReport, PrimitiveAction
from .anti_loop_critic import AntiLoopCritic
from .false_progress_detector import FalseProgressDetector
from .ontology_dissenter import OntologyDissenter


class DissentController:
    """Skeptical controller with authority to interrupt stale coherence."""

    def __init__(self) -> None:
        self.false_progress = FalseProgressDetector()
        self.ontology_dissenter = OntologyDissenter()
        self.loop_critic = AntiLoopCritic()
        self._last_interrupt_step = -999
        self._last_report = DissentReport()

    def update(self, obs, memory) -> DissentReport:
        chamber_warnings = {
            "physics": self._physics_warnings(memory),
            "strategy": self._strategy_warnings(memory),
            "composition": self._composition_warnings(memory),
            "memory": memory.game.bissociation_engine.warnings(memory),
        }
        report = DissentReport(
            false_progress=self.false_progress.analyze(memory),
            ontology_warnings=self.ontology_dissenter.analyze(memory),
            chamber_warnings=chamber_warnings,
            loop_warning=self.loop_critic.analyze(memory),
            suggested_actions=[],
        )
        if report.loop_warning:
            report.suggested_actions.append("diversify")
        if report.false_progress.get("high_sp_low_tp", 0.0) > 0.30:
            report.suggested_actions.append("closure")
        if report.false_progress.get("high_lp_low_sp", 0.0) > 0.25:
            report.suggested_actions.append("experiment")
        if any(chamber_warnings.values()):
            report.suggested_actions.append("reframe")
        memory.fast.last_dissent = report
        self._last_report = report
        return report

    def should_interrupt(self, memory) -> bool:
        step = memory.game.total_actions
        if step - self._last_interrupt_step < 8:
            return False
        report = self._last_report
        if report.loop_warning:
            return True
        if report.false_progress.get("repeat_pressure", 0.0) > 0.45:
            return True
        if len(report.ontology_warnings) >= 2:
            return True
        if sum(len(items) for items in report.chamber_warnings.values()) >= 2:
            return True
        return False

    def interrupt_and_redirect(self, obs, memory) -> ActionIntent:
        self._last_interrupt_step = memory.game.total_actions
        report = self._last_report
        if memory.game.progress.should_kill_branch():
            memory.game.branch_scheduler.on_branch_kill(memory)
            return ActionIntent(
                source="dissent",
                primitive_plan=[PrimitiveAction("RESET")],
                metadata={"reason": "branch_kill"},
            )

        if report.ontology_warnings:
            top = memory.game.current_ontologies[0] if memory.game.current_ontologies else None
            if top is not None:
                memory.ontology_competition.downweight(top.ontology_id, amount=0.5)

        current_project_id = memory.fast.current_project_id
        if current_project_id and memory.game.project_market is not None:
            project = memory.game.project_market.projects.get(current_project_id)
            if project is not None:
                memory.game.project_market.suspend_family(project.kind, steps=18)

        least_tried = memory.game.profiler.least_tried_actions(obs.available_actions, k=3)
        if obs.objects and any(action == "ACTION6" for action in obs.available_actions):
            target = random.choice(obs.objects[: min(len(obs.objects), 4)])
            return ActionIntent(
                source="dissent",
                primitive_plan=[
                    PrimitiveAction(
                        "ACTION6",
                        x=int(round(target.center[1])),
                        y=int(round(target.center[0])),
                    )
                ],
                metadata={"reason": "skeptical_probe"},
            )

        primitive = PrimitiveAction(least_tried[0] if least_tried else "ACTION1")
        return ActionIntent(
            source="dissent",
            primitive_plan=[primitive],
            metadata={"reason": "diversify"},
        )

    def _physics_warnings(self, memory) -> list[str]:
        warnings: list[str] = []
        num_ops = len(memory.game.inducer.operators)
        pred = memory.game.inducer.operator_predictive_accuracy()
        if num_ops > 10:
            warnings.append("operator ecology exceeds budget")
        if num_ops >= 8 and pred < 0.35:
            warnings.append("operators are multiplying faster than they predict")
        if len(memory.game.teleology.speculative_hypotheses()) > len(memory.game.teleology.hypotheses()) + 1:
            warnings.append("teleology remains mostly speculative")
        return warnings

    def _strategy_warnings(self, memory) -> list[str]:
        warnings: list[str] = []
        projects = memory.game.selected_projects[-12:]
        if len(projects) >= 6:
            top = max(projects.count(project_id) for project_id in set(projects))
            if top / len(projects) > 0.58:
                warnings.append("one project family is monopolizing selection")
        if memory.game.project_market is not None and len(memory.game.project_market.projects) > 6:
            warnings.append("project market exceeds active budget")
        return warnings

    def _composition_warnings(self, memory) -> list[str]:
        warnings: list[str] = []
        if len(memory.game.motifs) > 5:
            warnings.append("motif inventory exceeds budget")
        if len(memory.game.rituals) > 2:
            warnings.append("ritual inventory exceeds budget")
        return warnings
