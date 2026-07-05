"""Bissociative retrieval engine for V4."""

from __future__ import annotations

from ..schemas import BissociativeProposal, Project


class BissociationEngine:
    """Retrieve one dominant frame, one distant frame, and propose hybrids."""

    def propose(self, obs, memory) -> list[BissociativeProposal]:
        query = memory.game.frame_memory.build_query(obs, memory)
        primary_frames = memory.game.frame_memory.retrieve_best(query, memory, limit=1)
        secondary_frames = memory.game.frame_memory.retrieve_distant_outcome_match(query, memory, limit=3)
        if not primary_frames or not secondary_frames:
            return []

        primary = primary_frames[0]
        query_embedding = None
        if getattr(memory.game, "learning", None) is not None:
            query_embedding = memory.game.learning.world_embedding.embed_query(query, obs, memory)
        proposals: list[BissociativeProposal] = []
        for secondary in secondary_frames:
            if secondary.frame_id == primary.frame_id:
                continue
            negative = memory.game.bridge_memory.retrieve_negative(primary.frame_id, secondary.frame_id)
            penalty = negative[0].penalty_weight if negative else 0.0
            for bridge in memory.game.bridge_memory.retrieve(primary.frame_id, secondary.frame_id)[:2]:
                project = self._project_from_bridge(bridge, obs, memory)
                if project is None:
                    continue
                learned_bridge = (
                    memory.game.learning.bridge_value.estimate(bridge, primary, secondary, memory)
                    if getattr(memory.game, "learning", None) is not None else bridge.utility_score
                )
                embedding_bridge = (
                    memory.game.learning.world_embedding.bridge_affinity(query_embedding, primary, secondary)
                    if query_embedding is not None and getattr(memory.game, "learning", None) is not None
                    else 0.0
                )
                evidence_compatibility = 0.5 * primary.reliability + 0.5 * secondary.reliability
                score = (
                    0.17 * bridge.novelty_score
                    + 0.18 * evidence_compatibility
                    + 0.17 * project.expected_structural_gain
                    + 0.17 * project.expected_terminal_gain
                    + 0.16 * learned_bridge
                    + 0.15 * embedding_bridge
                    - 0.30 * penalty
                    - 0.10 * project.estimated_cost
                )
                if score <= 0.10:
                    continue
                project.metadata["bridge_learned_value"] = round(learned_bridge, 3)
                project.metadata["embedding_bridge_affinity"] = round(embedding_bridge, 3)
                proposals.append(
                    BissociativeProposal(
                        proposal_id=f"{bridge.bridge_id}:{project.project_id}",
                        source_frame_a=primary.frame_id,
                        source_frame_b=secondary.frame_id,
                        tension_type=str(bridge.hybrid_hypothesis_template.get("tension_type", "hybrid")),
                        hybrid_project=project,
                        confidence=min(1.0, score),
                        anti_loop_warning=(
                            "similar frame crossing previously led to loops"
                            if penalty > 0.3 else None
                        ),
                    )
                )

        proposals.sort(key=lambda proposal: proposal.confidence, reverse=True)
        return proposals[:2]

    def warnings(self, memory) -> list[str]:
        current = getattr(memory.game, "current_frame", None)
        previous = getattr(memory.game, "previous_frame", None)
        if current is None or previous is None:
            return []
        negatives = memory.game.bridge_memory.retrieve_negative(previous.frame_id, current.frame_id)
        if negatives and negatives[0].penalty_weight > 0.25:
            return [f"recent frame crossing {previous.ontology_kind}->{current.ontology_kind} is risky"]
        return []

    def _project_from_bridge(self, bridge, obs, memory) -> Project | None:
        template = bridge.hybrid_hypothesis_template
        kind = str(template.get("project_kind", "closure_probe"))
        description = str(template.get("description", "Hybrid project"))
        ontology_id = memory.game.current_ontologies[0].ontology_id if memory.game.current_ontologies else "token_world"
        metadata = {
            "bissociative": True,
            "source_bridge": bridge.bridge_id,
            "source_frames": (bridge.frame_a_id, bridge.frame_b_id),
        }

        if kind == "probe_unique_object":
            uniques = [
                obj for obj in obs.objects
                if obj.value != 0 and sum(1 for other in obs.objects if other.value == obj.value) == 1
            ]
            if not uniques:
                return None
            target = min(uniques, key=lambda obj: obj.area)
            metadata.update(
                {
                    "target_value": target.value,
                    "target_pos": (int(round(target.center[0])), int(round(target.center[1]))),
                }
            )
            return Project(
                project_id=f"bridge_probe_{target.value}",
                description=description,
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="probe_unique_object",
                expected_info_gain=0.28,
                expected_structural_gain=0.20,
                expected_terminal_gain=0.20 + 0.20 * bridge.utility_score,
                estimated_cost=0.30,
                fragility=0.28,
                dignity=0.58,
                metadata=metadata,
            )

        if kind == "transform_then_probe":
            if not memory.game.inducer.get_by_kind("global_transform"):
                return None
            if obs.objects:
                target = min(obs.objects, key=lambda obj: obj.area)
                metadata.update(
                    {
                        "target_value": target.value,
                        "target_pos": (int(round(target.center[0])), int(round(target.center[1]))),
                    }
                )
            return Project(
                project_id=f"bridge_transform_probe_{bridge.bridge_id}",
                description=description,
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="transform_then_probe",
                expected_info_gain=0.24,
                expected_structural_gain=0.22,
                expected_terminal_gain=0.18 + 0.20 * bridge.utility_score,
                estimated_cost=0.35,
                fragility=0.25,
                dignity=0.55,
                metadata=metadata,
            )

        if kind == "exhaust_class":
            counts: dict[int, int] = {}
            for obj in obs.objects:
                if obj.value == 0:
                    continue
                counts[obj.value] = counts.get(obj.value, 0) + 1
            if not counts:
                return None
            target_value = min(counts, key=lambda value: counts[value])
            target = next((obj for obj in obs.objects if obj.value == target_value), None)
            if target is not None:
                metadata.update(
                    {
                        "target_value": target_value,
                        "target_pos": (int(round(target.center[0])), int(round(target.center[1]))),
                        "remaining": counts[target_value],
                    }
                )
            return Project(
                project_id=f"bridge_exhaust_{target_value}",
                description=description,
                ontology_id=ontology_id,
                law_dependencies=[],
                kind="exhaust_class",
                expected_info_gain=0.18,
                expected_structural_gain=0.28,
                expected_terminal_gain=0.16 + 0.18 * bridge.utility_score,
                estimated_cost=0.40,
                fragility=0.30,
                dignity=0.56,
                metadata=metadata,
            )

        return Project(
            project_id=f"bridge_closure_{bridge.bridge_id}",
            description=description,
            ontology_id=ontology_id,
            law_dependencies=[],
            kind="closure_probe",
            expected_info_gain=0.16,
            expected_structural_gain=0.12,
            expected_terminal_gain=0.18 + 0.20 * bridge.utility_score,
            estimated_cost=0.25,
            fragility=0.25,
            dignity=0.54,
            metadata=metadata,
        )
