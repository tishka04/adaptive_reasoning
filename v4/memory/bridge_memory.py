"""Bridge memory for bissociative retrieval."""

from __future__ import annotations

from ..schemas import BissociationBridge, MemoryFrame, NegativeBridge


class BridgeMemory:
    """Store useful and harmful crossings between different interpretive frames."""

    def __init__(self) -> None:
        self.bridges: dict[str, BissociationBridge] = {}
        self.negative_bridges: dict[str, NegativeBridge] = {}
        self._seed_manual_bridges()

    def _seed_manual_bridges(self) -> None:
        self.add_bridge(
            BissociationBridge(
                bridge_id="manual_avatar_click",
                frame_a_id="avatar_world",
                frame_b_id="click_world",
                trigger_signature={"reason": "movement_reaches_target_then_click"},
                hybrid_hypothesis_template={
                    "project_kind": "probe_unique_object",
                    "description": "Reach a target, then click it as a closure probe",
                    "tension_type": "move_then_click",
                },
                novelty_score=0.45,
                utility_score=0.30,
            )
        )
        self.add_bridge(
            BissociationBridge(
                bridge_id="manual_transform_click",
                frame_a_id="transform_world",
                frame_b_id="click_world",
                trigger_signature={"reason": "transform_unlocks_click"},
                hybrid_hypothesis_template={
                    "project_kind": "transform_then_probe",
                    "description": "Transform first, then probe objects immediately",
                    "tension_type": "transform_then_click",
                },
                novelty_score=0.40,
                utility_score=0.32,
            )
        )
        self.add_bridge(
            BissociationBridge(
                bridge_id="manual_avatar_token",
                frame_a_id="avatar_world",
                frame_b_id="token_world",
                trigger_signature={"reason": "navigation_reveals_count_goal"},
                hybrid_hypothesis_template={
                    "project_kind": "exhaust_class",
                    "description": "Use movement to reach and exhaust a target class",
                    "tension_type": "reach_then_exhaust",
                },
                novelty_score=0.35,
                utility_score=0.26,
            )
        )
        self.add_bridge(
            BissociationBridge(
                bridge_id="manual_structural_closure",
                frame_a_id="field_world",
                frame_b_id="click_world",
                trigger_signature={"reason": "structural_change_needs_final_probe"},
                hybrid_hypothesis_template={
                    "project_kind": "closure_probe",
                    "description": "After structural change, try a pointed closure probe",
                    "tension_type": "structure_then_close",
                },
                novelty_score=0.38,
                utility_score=0.24,
            )
        )

    def add_bridge(self, bridge: BissociationBridge) -> None:
        existing = self.bridges.get(bridge.bridge_id)
        if existing is None:
            self.bridges[bridge.bridge_id] = bridge
            return
        existing.success_count += bridge.success_count
        existing.failure_count += bridge.failure_count
        existing.novelty_score = max(existing.novelty_score, bridge.novelty_score)
        existing.utility_score = max(existing.utility_score, bridge.utility_score)

    def add_negative_bridge(self, bridge: NegativeBridge) -> None:
        existing = self.negative_bridges.get(bridge.bridge_id)
        if existing is None:
            self.negative_bridges[bridge.bridge_id] = bridge
            return
        existing.penalty_weight = max(existing.penalty_weight, bridge.penalty_weight)

    def observe_shift(
        self,
        previous_frame: MemoryFrame | None,
        current_frame: MemoryFrame | None,
        sp_gain: float,
        tp_gain: float,
        loop_warning: bool = False,
    ) -> None:
        if previous_frame is None or current_frame is None:
            return
        if previous_frame.frame_id == current_frame.frame_id:
            return

        pair_id = f"{previous_frame.ontology_kind}->{current_frame.ontology_kind}"
        if sp_gain + tp_gain > 0.04 and previous_frame.ontology_kind != current_frame.ontology_kind:
            project_kind = current_frame.dominant_projects[0] if current_frame.dominant_projects else "closure_probe"
            bridge = self.bridges.get(pair_id)
            if bridge is None:
                bridge = BissociationBridge(
                    bridge_id=pair_id,
                    frame_a_id=previous_frame.ontology_kind,
                    frame_b_id=current_frame.ontology_kind,
                    trigger_signature={"terminal_style": current_frame.terminal_style},
                    hybrid_hypothesis_template={
                        "project_kind": project_kind,
                        "description": f"Shift from {previous_frame.ontology_kind} to {current_frame.ontology_kind}",
                        "tension_type": "learned_shift",
                    },
                    novelty_score=0.25,
                    utility_score=0.0,
                )
                self.bridges[pair_id] = bridge
            bridge.success_count += 1
            bridge.utility_score = min(1.0, bridge.utility_score + 0.5 * sp_gain + tp_gain)

        if loop_warning and previous_frame.ontology_kind != current_frame.ontology_kind:
            negative = NegativeBridge(
                bridge_id=f"neg:{pair_id}",
                frame_a_id=previous_frame.ontology_kind,
                frame_b_id=current_frame.ontology_kind,
                failure_signature={"reason": "loop_after_shift"},
                penalty_weight=min(1.0, 0.20 + sp_gain * 0.2),
            )
            self.add_negative_bridge(negative)

    def retrieve(self, frame_a_id: str, frame_b_id: str) -> list[BissociationBridge]:
        matches = []
        for bridge in self.bridges.values():
            if (
                {bridge.frame_a_id, bridge.frame_b_id} == {frame_a_id, frame_b_id}
                or {bridge.frame_a_id, bridge.frame_b_id} == {
                    frame_a_id.split("|")[0],
                    frame_b_id.split("|")[0],
                }
            ):
                matches.append(bridge)
        matches.sort(
            key=lambda bridge: (
                bridge.utility_score,
                bridge.success_count - bridge.failure_count,
                bridge.novelty_score,
            ),
            reverse=True,
        )
        return matches

    def retrieve_negative(self, frame_a_id: str, frame_b_id: str) -> list[NegativeBridge]:
        matches = []
        for bridge in self.negative_bridges.values():
            if (
                {bridge.frame_a_id, bridge.frame_b_id} == {frame_a_id, frame_b_id}
                or {bridge.frame_a_id, bridge.frame_b_id} == {
                    frame_a_id.split("|")[0],
                    frame_b_id.split("|")[0],
                }
            ):
                matches.append(bridge)
        matches.sort(key=lambda bridge: bridge.penalty_weight, reverse=True)
        return matches
