# A39 milestones - precondition-aware policy rollout

Derniere mise a jour : 2026-06-19

A39 reinjecte le raffinement A38 dans le rollout ferme. La policy utilise une
mecanique confirmee seulement si le scope A35 couvre le contexte et si les
preconditions dynamiques A38 sont satisfaites.

## Principe fixe

A39 ne dit pas : "la mecanique est vraie".

A39 dit seulement :

- la mecanique est disponible dans A33 ;
- le scope A35 peut proposer son action ;
- A38 peut inhiber cette action si l'effet est deja consomme ;
- le rollout execute alors un fallback neutre.

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A39`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A39.1 - Precondition-aware policy rollout | Fait | `theory/a39/precondition_aware_policy_rollout.py`, `tests/test_a39_precondition_aware_policy_rollout.py`, `diagnostics/a39/precondition_aware_policy_rollout.json` | Reutilisation saturee evitee; contradiction d'usage supprimee; aucun verdict de verite |

## A39.1 - Precondition-aware policy rollout

Entrees :

- `diagnostics/a33/confirmed_mechanics_registry.json`
- `diagnostics/a35/confirmed_mechanic_scope_map.json`
- `diagnostics/a38/rollout_aware_scope_refinement.json`

Sortie :

- `diagnostics/a39/precondition_aware_policy_rollout.json`

Contrat :

- `RESET`
- Pour chaque step du budget :
  - proposer une action par scope A35 ;
  - si l'action vient d'une mecanique confirmee, verifier les preconditions A38 ;
  - bloquer `ACTION6` si `target_patch_not_already_saturated=false` ;
  - executer un fallback neutre en cas de precondition echouee ;
  - ne jamais modifier A33/A35/A38.

KPIs :

- `avoided_saturated_reuse`
- `usage_contradictions`
- `policy_steps_from_confirmed_mechanic`
- `fallback_due_to_failed_precondition`
- `wrong_confirmations=0`
- `truth_status=NOT_REEVALUATED_BY_A39`

Run du 2026-06-19 :

- entrees lues : A33, A35, A38
- artefacts non lus : M3, A32, A34, A36, A37
- artefacts non modifies : A33, A35, A38
- budget : 3 pas live
- `policy_steps=3`
- `policy_steps_from_confirmed_mechanic=1`
- `fallback_steps=2`
- `fallback_due_to_failed_precondition=1`
- `avoided_saturated_reuse=1`
- `blocked_confirmed_mechanic_steps=1`
- `functional_progress_steps=1`
- `useful_new_states=1`
- `usage_contradictions=0`
- `cycle_or_dead_end_detected=false`
- `truth_status=NOT_REEVALUATED_BY_A39`
- `revision_performed=false`
- `wrong_confirmations=0`

Rollout documente :

| Step | Contexte | Decision | Precondition | Action executee | Signal | Lecture |
|---:|---|---|---|---|---:|---|
| 0 | `reset_exact` | proposer `ACTION6` | `SATISFIED`, patch non sature | `ACTION6` | 1 | usage utile |
| 1 | `after_ACTION6` | contexte non couvert | `NOT_APPLICABLE` | `ACTION3` | 0 | fallback neutre |
| 2 | `after_ACTION3` | proposer `ACTION6` | `FAILED`, `target_patch_not_already_saturated=true` | `ACTION4` | 0 | reutilisation saturee evitee |

Details du blocage :

- contexte couvert par A35 : `after_ACTION3`
- contexte live raffine par A38 : `after_ACTION3_live_after_ACTION6`
- action bloquee : `ACTION6`
- patch courant : `[[5,5,5],[3,5,10]]`
- patch sature de reference : `[[5,5,5],[3,5,10]]`
- `target_patch_already_saturated=true`
- `avoided_saturated_reuse=true`

Lecture :

- A39 montre que l'agent apprend a ne pas reutiliser une competence consommee.
- Le meme suffixe `after_ACTION3` qui provoquait une contradiction en A37 est
  maintenant inhibe par A38 parce que le patch local est deja sature.
- La contradiction d'usage tombe de 1 dans A37 a 0 dans A39.
- Un fallback du a une precondition echouee est un comportement attendu, pas un
  echec du mecanisme.

## Commandes de verification

Tests A39 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a39_precondition_aware_policy_rollout.py -q
```

Run A39 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a39.precondition_aware_policy_rollout --registry diagnostics\a33\confirmed_mechanics_registry.json --scope-map diagnostics\a35\confirmed_mechanic_scope_map.json --refinement diagnostics\a38\rollout_aware_scope_refinement.json --budget 3 --out diagnostics\a39\precondition_aware_policy_rollout.json
```

Guard A39 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a39\*.py,diagnostics\a39\*.json,A39_MILESTONES.md -Pattern $patterns -SimpleMatch
```
