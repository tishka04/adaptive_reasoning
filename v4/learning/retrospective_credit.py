"""Retrospective credit assignment for V4-Learn."""

from __future__ import annotations

from .common import outcome_reward


class RetrospectiveCredit:
    """Turn agent attempts into training signals for the learning layer."""

    def __init__(self, learning) -> None:
        self.learning = learning

    def on_transition(self, memory, transition) -> None:
        lp_gain = float(transition.metadata.get("lp_delta", 0.0))
        sp_gain = float(transition.metadata.get("sp_delta", 0.0))
        tp_gain = float(transition.metadata.get("tp_delta", 0.0))
        loop_warning = bool(memory.fast.last_dissent and memory.fast.last_dissent.loop_warning)
        sterile_now = bool(memory.game.progress.should_kill_branch())
        reward = outcome_reward(
            lp_gain=lp_gain,
            sp_gain=sp_gain,
            tp_gain=tp_gain,
            solved=transition.level_completed,
            sterile=sterile_now,
            loop=loop_warning,
        )

        self.learning.world_reliability.update(
            memory,
            reward=reward,
            sterile=sterile_now,
            solved=transition.level_completed,
        )
        self.learning.sterility_predictor.update(memory, sterile=sterile_now, reward=reward)

        project_id = str(transition.metadata.get("project_id", "") or "")
        project_kind = str(transition.metadata.get("project_kind", "") or "")
        ontology_kind = str(transition.metadata.get("ontology_kind", "") or "")
        phase = str(transition.metadata.get("phase", memory.fast.current_phase))
        lp, sp, tp = memory.game.progress.scores()

        if project_id and memory.game.project_market is not None:
            project = memory.game.project_market.projects.get(project_id)
            if project is not None:
                self.learning.project_value.update(project, memory, reward=reward)
                project_kind = project.kind
        elif project_kind:
            self.learning.project_value.update_by_metadata(
                project_kind=project_kind,
                ontology_kind=ontology_kind or "unknown",
                phase=phase,
                lp=lp,
                sp=sp,
                tp=tp,
                reward=reward,
            )

        bandit_signature = transition.metadata.get("bandit_signature")
        if isinstance(bandit_signature, tuple):
            self.learning.arbiter_bandit.update(bandit_signature, reward)
        elif transition.metadata.get("source") and project_kind:
            self.learning.arbiter_bandit.update_by_metadata(
                mind_name=str(transition.metadata["source"]),
                project_kind=project_kind,
                ontology_kind=ontology_kind or "unknown",
                phase=phase,
                lp=lp,
                sp=sp,
                tp=tp,
                reward=reward,
            )

        if ontology_kind:
            self.learning.ontology_calibrator.update(ontology_kind, memory, reward=reward)

        operator_id = transition.metadata.get("operator_id")
        if operator_id:
            operator = memory.game.inducer.operators.get(str(operator_id))
            if operator is not None:
                self.learning.operator_utility.update(operator, memory, reward=reward)

        bridge_key = transition.metadata.get("source_bridge")
        if bridge_key:
            self.learning.bridge_value.update(
                bridge_key=str(bridge_key),
                phase=phase,
                sp=sp,
                tp=tp,
                reward=reward,
            )

        ritual_id = transition.metadata.get("ritual_id")
        if ritual_id and memory.game.rituals.get(str(ritual_id)) is not None:
            self.learning.compression_value.update_ritual(
                memory.game.rituals[str(ritual_id)],
                reward=reward,
            )

        for motif in memory.game.motifs.values():
            if motif.kind == "ontology_shift" or tp_gain > 0.01 or sp_gain > 0.01:
                self.learning.compression_value.update_motif(motif, reward=reward)

        for rule in memory.game.teleology.hypotheses() + memory.game.teleology.speculative_hypotheses():
            if memory.game.teleology._rule_matches(rule, transition, memory.fast.current_obs):
                self.learning.teleology_validator.update(rule, memory, reward=reward)

    def on_branch_reset(self, memory) -> None:
        self.learning.sterility_predictor.update(memory, sterile=True, reward=0.0)
        self.learning.world_reliability.update(memory, reward=0.0, sterile=True, solved=False)

    def on_game_end(self, memory, won: bool) -> None:
        self.learning.world_reliability.update(memory, reward=1.0 if won else 0.0, sterile=not won, solved=won)

