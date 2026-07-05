"""Project generation for V4's strategy market."""

from __future__ import annotations

from typing import Any

from ..schemas import ObservationV4, Project


class ProjectGenerator:
    """Generate compact candidate undertakings from the current world-model."""

    def generate(self, obs: ObservationV4, memory: Any) -> list[Project]:
        ranked_ontologies = getattr(memory.game, "current_ontologies", [])
        top_ontology = ranked_ontologies[0] if ranked_ontologies else None
        ontology_id = top_ontology.ontology_id if top_ontology is not None else "token_world"
        lp, sp, tp = memory.game.progress.scores()
        projects: dict[str, Project] = {}

        class_counts: dict[int, int] = {}
        class_positions: dict[int, tuple[int, int]] = {}
        for obj in obs.objects:
            class_counts[obj.value] = class_counts.get(obj.value, 0) + 1
            if obj.value not in class_positions:
                class_positions[obj.value] = (
                    int(round(obj.center[0])),
                    int(round(obj.center[1])),
                )

        for rule in memory.game.teleology.hypotheses()[:3]:
            if rule.effect.args.get("kind") == "class_exhaustion":
                value = int(rule.effect.args.get("value", -1))
                if value in class_counts:
                    project_id = f"exhaust_{value}"
                    projects[project_id] = Project(
                        project_id=project_id,
                        description=f"Exhaust class {value}",
                        ontology_id=ontology_id,
                        law_dependencies=[rule.rule_id],
                        kind="exhaust_class",
                        expected_info_gain=0.15,
                        expected_structural_gain=0.35,
                        expected_terminal_gain=min(0.65, 0.12 + 0.28 * rule.confidence),
                        estimated_cost=min(1.0, 0.15 * class_counts[value] + 0.2),
                        fragility=max(0.1, 0.5 - 0.2 * rule.confidence),
                        dignity=0.65,
                        metadata={
                            "target_value": value,
                            "target_pos": class_positions.get(value),
                            "remaining": class_counts[value],
                        },
                    )

        unique_values = [
            (value, pos)
            for value, pos in class_positions.items()
            if value != 0 and class_counts.get(value, 0) == 1
        ]
        for value, pos in unique_values[:4]:
            project_id = f"probe_{value}"
            projects[project_id] = Project(
                project_id=project_id,
                description=f"Probe unique object class {value}",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="probe_unique_object",
                expected_info_gain=0.45,
                expected_structural_gain=0.18,
                expected_terminal_gain=0.05 if tp < 0.12 else 0.14,
                estimated_cost=0.35,
                fragility=0.35,
                dignity=0.55,
                metadata={"target_value": value, "target_pos": pos},
            )

        for region_id in obs.topology.unlocked_regions[:3]:
            target = None
            if 0 <= region_id < len(obs.topology.reachable_regions):
                cells = list(obs.topology.reachable_regions[region_id])
                if cells:
                    target = max(cells, key=lambda cell: (cell[0], cell[1]))
            project_id = f"reach_region_{region_id}"
            projects[project_id] = Project(
                project_id=project_id,
                description=f"Reach region {region_id}",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="reach_region",
                expected_info_gain=0.25,
                expected_structural_gain=0.45,
                expected_terminal_gain=0.08,
                estimated_cost=0.45,
                fragility=0.25,
                dignity=0.60,
                metadata={"target_region": region_id, "target_pos": target},
            )

        if memory.game.inducer.get_by_kind("global_transform"):
            project_id = "transform_then_probe"
            projects[project_id] = Project(
                project_id=project_id,
                description="Trigger transform then immediately probe",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="transform_then_probe",
                expected_info_gain=0.35,
                expected_structural_gain=0.28,
                expected_terminal_gain=0.12 if tp > 0.20 else 0.04,
                estimated_cost=0.40,
                fragility=0.30,
                dignity=0.50,
            )

        ritual = memory.game.best_ritual_for(top_ontology.kind if top_ontology else "token_world")
        if ritual is not None:
            project_id = f"replay_{ritual.ritual_id}"
            projects[project_id] = Project(
                project_id=project_id,
                description=f"Replay ritual {ritual.ritual_id} then deviate",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="replay_prefix_then_deviate",
                expected_info_gain=0.18,
                expected_structural_gain=0.20,
                expected_terminal_gain=min(0.55, ritual.success_rate + 0.05),
                estimated_cost=min(1.0, len(ritual.prefix) / 12.0),
                fragility=0.25,
                dignity=0.70,
                metadata={"ritual_id": ritual.ritual_id},
            )

        if tp > 0.20 or sum(1 for obj in obs.objects if obj.value != 0 and obj.area <= 12) <= 3:
            project_id = "closure_probe"
            projects[project_id] = Project(
                project_id=project_id,
                description="Aggressively test terminal closure hypotheses",
                ontology_id=ontology_id,
                law_dependencies=[rule.rule_id for rule in memory.game.teleology.hypotheses()[:2]],
                kind="closure_probe",
                expected_info_gain=0.18,
                expected_structural_gain=0.12,
                expected_terminal_gain=min(0.60, max(0.18, tp + 0.08)),
                estimated_cost=0.35,
                fragility=0.30,
                dignity=0.75,
            )

        if sp < 0.18 or memory.game.progress.should_kill_branch():
            project_id = "sequence_combo_test"
            projects[project_id] = Project(
                project_id=project_id,
                description="Test short action motifs and deviations",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="sequence_combo_test",
                expected_info_gain=0.40,
                expected_structural_gain=0.16,
                expected_terminal_gain=0.05,
                estimated_cost=0.25,
                fragility=0.20,
                dignity=0.52,
            )

        if memory.game.progress.should_kill_branch():
            project_id = "escape_dead_region"
            projects[project_id] = Project(
                project_id=project_id,
                description="Escape sterile local regime",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="escape_dead_region",
                expected_info_gain=0.30,
                expected_structural_gain=0.22,
                expected_terminal_gain=0.06,
                estimated_cost=0.20,
                fragility=0.15,
                dignity=0.65,
            )

        if not projects:
            projects["bootstrap_probe"] = Project(
                project_id="bootstrap_probe",
                description="Bootstrap world understanding",
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="sequence_combo_test",
                expected_info_gain=0.45,
                expected_structural_gain=0.10,
                expected_terminal_gain=0.02,
                estimated_cost=0.20,
                fragility=0.18,
                dignity=0.50,
            )

        for proposal in memory.game.bissociation_engine.propose(obs, memory):
            if proposal.hybrid_project is None:
                continue
            project = proposal.hybrid_project
            project.metadata["bissociative_confidence"] = proposal.confidence
            projects[project.project_id] = project

        return list(projects.values())
