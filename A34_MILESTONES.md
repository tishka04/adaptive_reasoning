# A34 milestones - confirmed mechanic usage probe

Derniere mise a jour : 2026-06-19

A34 teste l'utilite agentique d'une mecanique confirmee par A33. Il ne
re-confirme pas la mecanique. Il mesure si la memoire confirmee change le choix
d'action et produit un signal fonctionnel local.

## Principe fixe

A34 ne dit pas : "la mecanique est vraie".

A34 dit seulement :

- `USEFUL`
- `CONTEXTUALLY_USEFUL`
- `NOT_USEFUL`
- `CONTEXTUALLY_NOT_USEFUL`

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A34`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A34.1 - Confirmed mechanic usage probe | Fait | `theory/a34/confirmed_mechanic_usage_probe.py`, `tests/test_a34_confirmed_mechanic_usage_probe.py`, `diagnostics/a34/confirmed_mechanic_usage_probe.json` | ACTION6 priorisee depuis A33; usage local utile observe |

## A34.1 - Confirmed mechanic usage probe

Entree :

- `diagnostics/a33/confirmed_mechanics_registry.json`

Sortie :

- `diagnostics/a34/confirmed_mechanic_usage_probe.json`

Contrat :

- Baseline : action choisie sans registre A33, via ordre neutre
  `ACTION3`, `ACTION4`, `ACTION1`, `ACTION2`.
- Treatment : action issue du registre A33.
- Meme metrique que la mecanique confirmee : `local_patch_before_after`.
- A34 ne relit pas les artefacts M3.
- A34 ne modifie pas A33.
- A34 ne produit aucun verdict de verite.

Mesures :

- `action_choice_changed`
- `action_prioritized_from_registry`
- `local_patch_before_after_observed`
- `useful_new_state`
- `functional_progress`
- `contradiction`
- `game_score_unchanged_or_improved`

Run du 2026-06-19 :

- mecanique testee :
  `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- baseline : `ACTION3`
- treatment : `ACTION6`
- baseline `local_changed_pixels=0`
- treatment `local_changed_pixels=1`
- `utility_assessment=USEFUL`
- `action_choice_changed=true`
- `action_prioritized_from_registry=true`
- `local_patch_before_after_observed=true`
- `useful_new_state=true`
- `functional_progress=true`
- `contradiction=false`
- `truth_status=NOT_REEVALUATED_BY_A34`
- `wrong_confirmations=0`

Lecture :

- La mecanique confirmee A33 devient une competence locale minimale : elle
  change le choix d'action et produit le signal local attendu.
- Ce n'est pas encore une preuve de resolution du jeu.
- Ce n'est pas une nouvelle confirmation de verite.
- Le scope reste `local_context`.

## Commandes de verification

Tests A34 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a34_confirmed_mechanic_usage_probe.py -q
```

Run A34 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a34.confirmed_mechanic_usage_probe --registry diagnostics\a33\confirmed_mechanics_registry.json --out diagnostics\a34\confirmed_mechanic_usage_probe.json
```
