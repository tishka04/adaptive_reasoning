"""Ontology skepticism for V4."""

from __future__ import annotations


class OntologyDissenter:
    """Warn when one worldview is monopolizing despite poor returns."""

    def analyze(self, memory) -> list[str]:
        warnings: list[str] = []
        ontologies = getattr(memory.game, "current_ontologies", [])
        if not ontologies:
            return warnings
        top = ontologies[0]
        lp, sp, tp = memory.game.progress.scores()
        obs = memory.fast.current_obs

        if top.confidence > 0.72 and tp < 0.08 and memory.game.progress.state.terminal_stall_steps > 30:
            warnings.append(f"{top.kind} overconfident with weak terminal progress")
        if top.kind == "avatar_world" and (obs is None or obs.best_player is None or obs.best_player.confidence < 0.25):
            warnings.append("avatar ontology dominant but player identity is unstable")
        if top.kind == "click_world":
            recent_clicks = [
                transition for transition in list(memory.fast.recent_transitions)[-12:]
                if transition.action.x is not None
            ]
            useful_clicks = [
                transition for transition in recent_clicks if transition.diff.num_changed > 0
            ]
            if recent_clicks and len(useful_clicks) / len(recent_clicks) < 0.35:
                warnings.append("click ontology dominant but clicks rarely matter")
        if top.kind == "transform_world" and sp < 0.12 and tp < 0.06:
            warnings.append("transform ontology active without structural payoff")
        return warnings
