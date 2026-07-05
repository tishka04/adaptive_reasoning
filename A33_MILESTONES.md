# A33 milestones - confirmed mechanic registry

Derniere mise a jour : 2026-06-19

A33 separe la memoire scientifique confirmee des files de decision. Il ne relit
pas M3. Il consomme uniquement les decisions A32.

## Principe fixe

A33 ne confirme rien lui-meme. Il enregistre seulement les decisions A32 dont :

- `decision=REVISION_ACCEPTED_AS_CONFIRMED`
- `decision_record.status=confirmed`

A33 exclut explicitement :

- support reutilise comme support independant
- support trace
- support prior
- candidats unresolved

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A33.1 - Confirmed Mechanic Registry | Fait | `theory/a33/confirmed_mechanics_registry.py`, `tests/test_a33_confirmed_mechanics_registry.py`, `diagnostics/a33/confirmed_mechanics_registry.json` | 1 mecanique confirmee locale issue de A32 |

## A33.1 - Confirmed Mechanic Registry

Entree :

- `diagnostics/a32/a15_revision_decisions.json`

Sortie :

- `diagnostics/a33/confirmed_mechanics_registry.json`

Regle de selection :

- Ne consommer que `decision=REVISION_ACCEPTED_AS_CONFIRMED`.
- Ignorer `FOLLOWUP_REQUIRED`.
- Ignorer `REVISION_REJECTED_AS_INSUFFICIENT`.
- Ignorer tout `decision_record` qui n'est pas `status=confirmed`.

Champs conserves :

- `key`
- `game_id`
- `action`
- `mechanic_family`
- `predicted_metric`
- `confirmed_support_independent`
- `experiments_spent`
- `control_actions_used`
- `known_scope=local_context`

Run du 2026-06-19 :

- `confirmed_mechanics=1`
- `confirmed_support_independent_total=2`
- `experiments_spent_total=3`
- `reused_control_support_excluded_total=1`
- `known_scope=local_context`
- `trace_support_counted_as_proof=false`
- `prior_counted_as_proof=false`
- `unresolved_candidates_excluded=true`

Mecanique inscrite :

- `key=mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- `game_id=bp35-0a0ad940`
- `action=ACTION6`
- `mechanic_family=position_effect_candidate`
- `predicted_metric=local_patch_before_after`
- `confirmed_support_independent=2`
- `experiments_spent=3`
- `control_actions_used=[ACTION3, ACTION4]`
- `known_scope=local_context`

Lecture :

- A32 a confirme une mecanique locale.
- A33 l'extrait dans une memoire scientifique confirmee.
- A33 ne prouve pas encore que cette mecanique aide a progresser dans bp35.
- Le registre est pret a etre consomme par un futur planner ou par un adaptateur
  polymorphe, sans relire les artefacts M3.

## Commandes de verification

Tests A33 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a33_confirmed_mechanics_registry.py -q
```

Generation registre :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a33.confirmed_mechanics_registry --decisions diagnostics\a32\a15_revision_decisions.json --out diagnostics\a33\confirmed_mechanics_registry.json
```
