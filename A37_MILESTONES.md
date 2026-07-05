# A37 milestones - closed-loop policy rollout

Derniere mise a jour : 2026-06-19

A37 transforme la decision ponctuelle A36 en micro-rollout ferme. Un meme
environnement reste vivant pendant plusieurs pas ; la policy utilise seulement
A33 et A35 pour choisir entre action confirmee et fallback neutre.

## Principe fixe

A37 ne dit pas : "la mecanique est vraie".

A37 dit seulement :

- la policy peut utiliser une mecanique confirmee dans plusieurs pas ;
- certains pas viennent de la memoire A33+A35 ;
- certains pas retombent sur fallback neutre/exploration ;
- les effets d'usage restent utiles, contradictoires, cycliques ou neutres.

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A37`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A37.1 - Closed-loop policy rollout | Fait | `theory/a37/scope_conditioned_policy_rollout.py`, `tests/test_a37_scope_conditioned_policy_rollout.py`, `diagnostics/a37/scope_conditioned_policy_rollout.json` | 4 pas bp35; 2 pas depuis mecanique confirmee; 1 progres utile; 1 contradiction d'usage |

## A37.1 - Closed-loop policy rollout

Entrees :

- `diagnostics/a33/confirmed_mechanics_registry.json`
- `diagnostics/a35/confirmed_mechanic_scope_map.json`

Sortie :

- `diagnostics/a37/scope_conditioned_policy_rollout.json`

Contrat :

- `RESET`
- Pour chaque step du budget :
  - lire la frame live ;
  - extraire une `context_signature` courte depuis l'historique d'actions ;
  - si un suffixe de contexte est couvert par A35, prioriser l'action confirmee ;
  - sinon utiliser une baseline neutre / exploration courte ;
  - executer l'action dans le meme environnement vivant ;
  - mesurer l'effet local et les signaux d'usage.

KPIs :

- `policy_steps_from_confirmed_mechanic`
- `functional_progress_steps`
- `useful_new_states`
- `usage_contradictions`
- `fallback_steps`
- `repeated_usefulness`
- `cycle_or_dead_end_detected`

Run du 2026-06-19 :

- entrees lues : A33 et A35 uniquement
- artefacts non lus pour la decision : M3, A32, A34, A36
- budget : 4 pas live
- mecanique policy :
  `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- `policy_steps=4`
- `policy_steps_from_confirmed_mechanic=2`
- `fallback_steps=2`
- `functional_progress_steps=1`
- `useful_new_states=1`
- `usage_contradictions=1`
- `repeated_usefulness=0`
- `cycle_or_dead_end_detected=false`
- `dead_end_or_cycle_steps=0`
- `truth_status=NOT_REEVALUATED_BY_A37`
- `revision_performed=false`
- `wrong_confirmations=0`

Rollout documente :

| Step | Context signature | Action | Source | Signal | Progres | Contradiction | Lecture |
|---:|---|---|---|---:|---|---|---|
| 0 | `reset_exact` | `ACTION6` | mecanique confirmee A33 + scope A35 | 1 | true | false | premier usage utile |
| 1 | `after_ACTION6` | `ACTION3` | fallback neutre | 0 | false | false | contexte non couvert |
| 2 | `after_ACTION3` | `ACTION6` | mecanique confirmee A33 + scope A35 | 0 | false | true | suffixe couvert mais etat live deja modifie par ACTION6 |
| 3 | `after_ACTION3_then_ACTION6` | `ACTION4` | fallback neutre | 0 | false | false | contexte non couvert |

Lecture :

- A37 ne cherche pas a gagner bp35.
- A37 teste si la competence locale peut etre reutilisee dans une trajectoire
  fermee.
- Le premier usage de `ACTION6` confirme l'utilite motrice deja observee par
  A36 dans un rollout vivant.
- Le deuxieme usage expose une limite importante : le suffixe `after_ACTION3`
  est couvert par A35 quand il est rejoue depuis `RESET`, mais il n'est pas
  equivalent apres une trajectoire vivante contenant deja `ACTION6`.
- Cette contradiction d'usage n'est pas une refutation de la mecanique A32/A33.
  C'est un signal que le scope A35 doit etre enrichi par des contextes
  rollout-aware.
- Les echecs ou cycles A37 sont des signaux de politique/scope, pas des verdicts
  scientifiques.

## Commandes de verification

Tests A37 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a37_scope_conditioned_policy_rollout.py -q
```

Run A37 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a37.scope_conditioned_policy_rollout --registry diagnostics\a33\confirmed_mechanics_registry.json --scope-map diagnostics\a35\confirmed_mechanic_scope_map.json --budget 4 --out diagnostics\a37\scope_conditioned_policy_rollout.json
```

Guard A37 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a37\*.py,diagnostics\a37\*.json,A37_MILESTONES.md -Pattern $patterns -SimpleMatch
```
