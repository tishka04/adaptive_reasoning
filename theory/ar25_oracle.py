"""Ground-truth MechanicsOracle for ar25, built from existing artefacts.

No environment replay is needed: the empirical action ontology
(``diagnostics/action_ontology/...``) and the human-compiled task program
(``task_programs/ar25.json``) already encode what each action really does and
which mechanics the human's winning trace relied on.

Fact namespaces produced:
  - ``action_effect::ACTIONn::<kind>``  empirical, true (dominant operator)
                                        and a few data-grounded FALSE claims
  - ``action_role::ACTIONn::<role>``    human-demonstrated, true
  - ``correspondence::ACTIONn::...``    relation-level validation facts
  - ``goal_family::<family>``           human-demonstrated, true
  - ``win_rule::<name>``                from anti-patterns, human-known FALSE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .epistemic_metrics import (
    GroundTruthFact,
    MechanicsOracle,
    mechanic_key,
    normalize_operator_kind,
)
from .correspondence_hypothesis import correspondence_key, normalize_pair_colors
from .role_hypotheses import action_role_key, goal_family_key, normalize_action_role

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ONTOLOGY = (
    _REPO_ROOT
    / "diagnostics"
    / "action_ontology"
    / "ar25-e3c63847.action_ontology.json"
)
_DEFAULT_TASK_PROGRAM = _REPO_ROOT / "task_programs" / "ar25.json"
_DEFAULT_ACTION2_SUCCESS = (
    _REPO_ROOT
    / "diagnostics"
    / "rule_inference"
    / "ar25-e3c63847.action2_success_condition.json"
)

# A clear state change; below this an action claiming "noop" might be defensible.
_CHANGE_THRESHOLD = 50.0


def build_ar25_oracle(
    action_ontology_path: Optional[Path] = None,
    task_program_path: Optional[Path] = None,
    game_id: str = "ar25-e3c63847",
) -> MechanicsOracle:
    """Assemble the ar25 ground-truth oracle from on-disk artefacts."""
    oracle = MechanicsOracle(game_id=game_id)
    _load_action_ontology(oracle, action_ontology_path or _DEFAULT_ONTOLOGY)
    _load_task_program(oracle, task_program_path or _DEFAULT_TASK_PROGRAM)
    _load_correspondence_diagnostics(oracle, _DEFAULT_ACTION2_SUCCESS)
    return oracle


def _load_action_ontology(oracle: MechanicsOracle, path: Path) -> None:
    if not Path(path).is_file():
        return
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    ontology = data.get("action_ontology", data)

    for action, info in ontology.items():
        if not str(action).upper().startswith("ACTION"):
            continue
        raw_dominant = normalize_operator_kind(info.get("dominant_operator_type", ""))
        avg_changed = float(info.get("avg_changed_cells", 0.0) or 0.0)
        low_amplitude_transform = (
            raw_dominant == "global_transform"
            and 0.0 < avg_changed < _CHANGE_THRESHOLD
        )
        dominant = "click" if low_amplitude_transform else raw_dominant

        # TRUE: the dominant operator class observed for this action.
        oracle.add(GroundTruthFact(
            key=mechanic_key(action, dominant),
            truth_value=True,
            description=f"{action} behaves as {dominant}",
            source="action_ontology",
        ))

        # FALSE (data-grounded): ar25 has no avatar movement — claiming any
        # action moves a player avatar is false.
        if dominant != "move":
            oracle.add(GroundTruthFact(
                key=mechanic_key(action, "move"),
                truth_value=False,
                description=f"{action} does NOT move a player avatar",
                source="action_ontology",
            ))

        # FALSE: small selector/control changes are not global transforms even
        # when the empirical ontology's coarse label says transform_like.
        if low_amplitude_transform:
            oracle.add(GroundTruthFact(
                key=mechanic_key(action, "global_transform"),
                truth_value=False,
                description=(
                    f"{action} changes only ~{avg_changed:.0f} cells; "
                    "treat as local selector/click effect"
                ),
                source="action_ontology",
            ))

        # FALSE: an action that clearly changes the grid is not a no-op.
        if avg_changed >= _CHANGE_THRESHOLD and dominant != "noop":
            oracle.add(GroundTruthFact(
                key=mechanic_key(action, "noop"),
                truth_value=False,
                description=f"{action} changes the grid (~{avg_changed:.0f} cells)",
                source="action_ontology",
            ))


def _load_task_program(oracle: MechanicsOracle, path: Path) -> None:
    if not Path(path).is_file():
        return
    with open(path, "r", encoding="utf-8") as handle:
        program = json.load(handle)

    family = str(program.get("goal_family", "")).strip().lower()
    if family:
        oracle.add(GroundTruthFact(
            key=goal_family_key(family),
            truth_value=True,
            description=f"goal family is {family}",
            demonstrated_by_human=True,
            source="task_program",
        ))

    for role in program.get("action_roles", []):
        action = str(role.get("action", "")).upper()
        role_name = normalize_action_role(str(role.get("role", "")))
        if action and role_name:
            oracle.add(GroundTruthFact(
                key=action_role_key(action, role_name),
                truth_value=True,
                description=f"{action} plays the role {role_name}",
                demonstrated_by_human=True,
                source="task_program",
            ))

    # Anti-patterns encode mechanics the human KNOWS to be false.
    for anti in program.get("anti_patterns", []):
        text = str(anti).lower()
        if "all shapes" in text and "match" in text:
            oracle.add(GroundTruthFact(
                key="win_rule::all_shapes_must_match",
                truth_value=False,
                description="win does NOT require all shapes to match",
                demonstrated_by_human=True,
                source="task_program",
            ))


def _load_correspondence_diagnostics(oracle: MechanicsOracle, path: Path) -> None:
    """Add ar25's explicit source-target validation fact from diagnostics."""
    if not Path(path).is_file():
        return
    with open(path, "r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)

    pair_colors = normalize_pair_colors(diagnostics.get("pair_colors", (10, 11)))
    level_transitions = diagnostics.get("level_transitions", []) or []
    action2_successes = [
        transition for transition in level_transitions
        if str(transition.get("action_that_caused", "")).upper() == "ACTION2"
    ]
    if len(action2_successes) < 2:
        return

    first, second = pair_colors
    oracle.add(GroundTruthFact(
        key=correspondence_key("ACTION2", "validates", pair_colors),
        truth_value=True,
        description=(
            f"ACTION2 validates the {first}/{second} correspondence when "
            "the prepared state advances"
        ),
        demonstrated_by_human=False,
        source="action2_success_condition",
    ))


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    oracle = build_ar25_oracle()
    print(f"ar25 oracle: {len(oracle.facts)} facts")
    for key, fact in sorted(oracle.facts.items()):
        flag = "T" if fact.truth_value else "F"
        human = " [human]" if fact.demonstrated_by_human else ""
        print(f"  [{flag}] {key}{human}  ({fact.source})")
