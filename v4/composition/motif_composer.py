"""Motif extraction and promotion for V4."""

from __future__ import annotations

from ..schemas import Motif


class MotifComposer:
    """Promote reusable motifs from recent transitions."""

    def update(self, transitions, memory) -> None:
        recent = list(transitions)[-4:]
        if len(recent) < 2:
            return

        sp_gain = float(recent[-1].metadata.get("sp_delta", 0.0))
        tp_gain = float(recent[-1].metadata.get("tp_delta", 0.0))
        if sp_gain <= 0.0 and tp_gain <= 0.0:
            return

        action_seq = tuple(transition.action.name for transition in recent[-3:])
        motif = Motif(
            motif_id=f"action_seq:{','.join(action_seq)}",
            kind="action_seq",
            content={"actions": list(action_seq)},
            support=1,
            utility=sp_gain + tp_gain,
            terminal_association=tp_gain,
            structural_association=sp_gain,
        )
        memory.game.add_motif(motif)

        removed_values = recent[-1].metadata.get("removed_values", {})
        if removed_values:
            motif = Motif(
                motif_id="count_pattern:" + ",".join(f"{k}:{v}" for k, v in sorted(removed_values.items())),
                kind="count_pattern",
                content={"removed_values": dict(removed_values)},
                support=1,
                utility=0.5 * sp_gain + tp_gain,
                terminal_association=tp_gain,
                structural_association=sp_gain,
            )
            memory.game.add_motif(motif)

        ontologies = memory.game.ontology_history
        if len(ontologies) >= 2 and ontologies[-1][0][0] != ontologies[-2][0][0]:
            motif = Motif(
                motif_id=f"ontology_shift:{ontologies[-2][0][0]}->{ontologies[-1][0][0]}",
                kind="ontology_shift",
                content={"from": ontologies[-2][0][0], "to": ontologies[-1][0][0]},
                support=1,
                utility=0.25 + sp_gain + tp_gain,
                terminal_association=tp_gain,
                structural_association=sp_gain,
            )
            memory.game.add_motif(motif)
