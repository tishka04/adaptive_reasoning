"""Project-conditioned specialist minds for V4."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schemas import MindProposal, PrimitiveAction, Project


def _movement_ops(memory: Any):
    return [
        op
        for op in memory.game.inducer.get_by_kind("move")
        if op.risk_estimate < 0.7
    ]


def _click_ops(memory: Any):
    return memory.game.inducer.get_by_kind("click")


def _transform_ops(memory: Any):
    return memory.game.inducer.get_by_kind("global_transform")


class SpecialistMind(ABC):
    name = "base"

    @abstractmethod
    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        raise NotImplementedError


class NavigatorMind(SpecialistMind):
    name = "navigator"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        player = obs.best_player
        move_ops = _movement_ops(memory)
        if player is None or not move_ops:
            return []

        target = project.metadata.get("target_pos")
        if target is None and obs.objects:
            target_obj = min(
                obs.objects,
                key=lambda obj: abs(player.position[0] - round(obj.center[0]))
                + abs(player.position[1] - round(obj.center[1])),
            )
            target = (int(round(target_obj.center[0])), int(round(target_obj.center[1])))
        if target is None:
            return []

        plan: list[str] = []
        pr, pc = player.position
        tr, tc = target
        for _ in range(min(10, abs(tr - pr) + abs(tc - pc))):
            best_op = None
            best_score = -999.0
            for operator in move_ops:
                dy = int(operator.parameters.get("dy", 0))
                dx = int(operator.parameters.get("dx", 0))
                new_dist = abs(tr - (pr + dy)) + abs(tc - (pc + dx))
                old_dist = abs(tr - pr) + abs(tc - pc)
                score = (old_dist - new_dist) * operator.confidence - 0.4 * operator.risk_estimate
                if score > best_score:
                    best_score = score
                    best_op = operator
            if best_op is None or best_score <= 0:
                break
            plan.append(best_op.operator_id)
            pr += int(best_op.parameters.get("dy", 0))
            pc += int(best_op.parameters.get("dx", 0))
            if (pr, pc) == (tr, tc):
                break

        if not plan:
            return []
        distance = abs(tr - player.position[0]) + abs(tc - player.position[1])
        return [
            MindProposal(
                mind_name=self.name,
                project=project,
                operator_plan=plan,
                confidence=min(0.95, 0.45 + 0.05 * len(plan)),
                expected_lp=0.40,
                expected_sp=0.22 if project.kind == "reach_region" else 0.14,
                expected_tp=0.18 if project.kind == "closure_probe" else 0.08,
                cost=min(1.0, distance / 10.0),
                risk=min(1.0, sum(
                    memory.game.inducer.operators[op_id].risk_estimate for op_id in plan
                )),
                metadata={"target_pos": target},
            )
        ]


class ClickMind(SpecialistMind):
    name = "click"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        proposals: list[MindProposal] = []
        click_ops = _click_ops(memory)
        target_value = project.metadata.get("target_value")
        targets = [
            obj for obj in obs.objects
            if obj.value != 0 and (target_value is None or obj.value == target_value)
        ]
        if not targets:
            return []

        coords = [
            PrimitiveAction("ACTION6", x=int(round(obj.center[1])), y=int(round(obj.center[0])))
            for obj in targets[:6]
        ]
        proposals.append(
            MindProposal(
                mind_name=self.name,
                project=project,
                primitive_plan=coords,
                confidence=0.35 + 0.10 * min(3, len(coords)),
                expected_lp=0.10,
                expected_sp=0.30 if project.kind in {"probe_unique_object", "exhaust_class"} else 0.18,
                expected_tp=0.22 if project.kind in {"exhaust_class", "closure_probe"} else 0.08,
                cost=min(1.0, len(coords) / 8.0),
                risk=0.10,
            )
        )

        if click_ops:
            op = max(click_ops, key=lambda item: item.confidence)
            proposals.append(
                MindProposal(
                    mind_name=self.name,
                    project=project,
                    operator_plan=[op.operator_id],
                    primitive_plan=coords[:2],
                    confidence=min(0.95, op.confidence),
                    expected_lp=0.08,
                    expected_sp=0.20,
                    expected_tp=0.16,
                    cost=0.20,
                    risk=op.risk_estimate,
                )
            )
        return proposals


class SequenceMind(SpecialistMind):
    name = "sequence"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        proposals: list[MindProposal] = []
        ritual = None
        if project.kind == "replay_prefix_then_deviate":
            ritual_id = project.metadata.get("ritual_id")
            ritual = memory.game.rituals.get(ritual_id)
        if ritual is None:
            top_ontology = memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "token_world"
            ritual = memory.game.best_ritual_for(top_ontology)
        if ritual is not None:
            replay = ritual.prefix[:]
            if replay:
                proposals.append(
                    MindProposal(
                        mind_name=self.name,
                        project=project,
                        primitive_plan=replay + [PrimitiveAction("ACTION1")],
                        confidence=min(0.95, 0.45 + 0.4 * ritual.success_rate),
                        expected_lp=0.12,
                        expected_sp=0.15,
                        expected_tp=0.30,
                        cost=min(1.0, len(replay) / 10.0),
                        risk=0.12,
                        metadata={"ritual_id": ritual.ritual_id},
                    )
                )

        least_tried = memory.game.profiler.least_tried_actions(obs.available_actions, k=3)
        if least_tried:
            combo = [PrimitiveAction(name) for name in (least_tried[:2] + least_tried[:1])]
            proposals.append(
                MindProposal(
                    mind_name=self.name,
                    project=project,
                    primitive_plan=combo,
                    confidence=0.28,
                    expected_lp=0.08,
                    expected_sp=0.14,
                    expected_tp=0.04,
                    cost=0.20,
                    risk=0.15,
                )
            )
        return proposals


class PhysicsMind(SpecialistMind):
    name = "physics"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        player = obs.best_player
        move_ops = _movement_ops(memory)
        if player is None or not move_ops:
            return []

        grid = obs.raw_grid
        pr, pc = player.position
        pushes = []
        for operator in move_ops:
            dy = int(operator.parameters.get("dy", 0))
            dx = int(operator.parameters.get("dx", 0))
            nr, nc = pr + dy, pc + dx
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != 0:
                pushes.append(operator)

        proposals: list[MindProposal] = []
        if pushes:
            best = max(pushes, key=lambda item: item.confidence)
            proposals.append(
                MindProposal(
                    mind_name=self.name,
                    project=project,
                    operator_plan=[best.operator_id],
                    confidence=max(0.3, best.confidence),
                    expected_lp=0.25,
                    expected_sp=0.18,
                    expected_tp=0.06,
                    cost=0.18,
                    risk=best.risk_estimate,
                )
            )
        escape_ops = memory.game.inducer.get_by_kind("escape")
        if escape_ops and project.kind == "escape_dead_region":
            best = max(escape_ops, key=lambda item: item.confidence)
            proposals.append(
                MindProposal(
                    mind_name=self.name,
                    project=project,
                    operator_plan=[best.operator_id],
                    confidence=min(0.95, best.confidence + 0.1),
                    expected_lp=0.30,
                    expected_sp=0.12,
                    expected_tp=0.03,
                    cost=0.10,
                    risk=best.risk_estimate,
                )
            )
        return proposals


class TransformMind(SpecialistMind):
    name = "transform"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        transform_ops = _transform_ops(memory)
        if not transform_ops:
            return []
        best = max(transform_ops, key=lambda item: item.confidence)
        primitive_tail = []
        if obs.objects:
            obj = min(obs.objects, key=lambda item: item.area)
            primitive_tail = [
                PrimitiveAction("ACTION6", x=int(round(obj.center[1])), y=int(round(obj.center[0])))
            ]
        return [
            MindProposal(
                mind_name=self.name,
                project=project,
                operator_plan=[best.operator_id],
                primitive_plan=primitive_tail,
                confidence=min(0.95, best.confidence),
                expected_lp=0.08,
                expected_sp=0.26,
                expected_tp=0.18 if project.kind == "transform_then_probe" else 0.08,
                cost=0.22,
                risk=best.risk_estimate,
            )
        ]


class ClosureMind(SpecialistMind):
    name = "closure"

    def propose(self, obs, memory: Any, project: Project) -> list[MindProposal]:
        small_objects = [
            obj for obj in obs.objects if obj.value != 0 and obj.area <= 12
        ]
        if not small_objects:
            return []
        plan = [
            PrimitiveAction("ACTION6", x=int(round(obj.center[1])), y=int(round(obj.center[0])))
            for obj in small_objects[:6]
        ]
        if obs.best_player is not None and _movement_ops(memory):
            target = small_objects[0]
            nav = NavigatorMind().propose(obs, memory, Project(
                project_id=f"{project.project_id}_nav",
                description="closure nav",
                ontology_id=project.ontology_id,
                law_dependencies=[],
                kind="reach_region",
                expected_info_gain=0.1,
                expected_structural_gain=0.1,
                expected_terminal_gain=0.2,
                estimated_cost=0.2,
                fragility=0.2,
                dignity=0.2,
                metadata={"target_pos": (int(round(target.center[0])), int(round(target.center[1])))},
            ))
            if nav:
                return [
                    MindProposal(
                        mind_name=self.name,
                        project=project,
                        operator_plan=nav[0].operator_plan,
                        primitive_plan=plan[:2],
                        confidence=min(0.95, 0.55 + 0.05 * len(plan)),
                        expected_lp=0.20,
                        expected_sp=0.15,
                        expected_tp=0.35,
                        cost=min(1.0, len(plan) / 8.0),
                        risk=0.15,
                    )
                ]
        return [
            MindProposal(
                mind_name=self.name,
                project=project,
                primitive_plan=plan,
                confidence=0.55,
                expected_lp=0.08,
                expected_sp=0.12,
                expected_tp=0.40,
                cost=min(1.0, len(plan) / 8.0),
                risk=0.16,
            )
        ]


def create_default_minds() -> list[SpecialistMind]:
    return [
        NavigatorMind(),
        ClickMind(),
        SequenceMind(),
        PhysicsMind(),
        TransformMind(),
        ClosureMind(),
    ]
