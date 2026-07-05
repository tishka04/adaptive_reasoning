# A38 milestones - rollout-aware scope refiner

Derniere mise a jour : 2026-06-19

A38 raffine le scope A35 avec les observations live A37. Il ne modifie pas la
verite scientifique A32/A33. Il apprend seulement des preconditions d'usage
dynamiques pour eviter de reutiliser aveuglement une competence deja consommee.

## Principe fixe

A38 ne dit pas : "ACTION6 est faux".

A38 dit seulement :

- ACTION6 reste une mecanique confirmee par A32/A33 ;
- son usage moteur peut etre bloque dans certains etats live ;
- le scope doit integrer une precondition d'effet non sature ;
- A33 n'est pas modifie.

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A38`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A38.1 - Rollout-aware scope refinement | Fait | `theory/a38/rollout_aware_scope_refinement.py`, `tests/test_a38_rollout_aware_scope_refinement.py`, `diagnostics/a38/rollout_aware_scope_refinement.json` | Saturation detectee; scope raffine avec preconditions; aucun verdict de verite |

## A38.1 - Rollout-aware scope refinement

Entrees :

- `diagnostics/a33/confirmed_mechanics_registry.json`
- `diagnostics/a35/confirmed_mechanic_scope_map.json`
- `diagnostics/a37/scope_conditioned_policy_rollout.json`

Sortie :

- `diagnostics/a38/rollout_aware_scope_refinement.json`

Contrat :

- Extraire les usages positifs A37.
- Extraire les usages negatifs A37.
- Detecter si le patch cible est deja sature.
- Produire un scope raffine d'usage.
- Ne jamais changer la verite de la mecanique.
- Ne jamais modifier A33.

Sorties attendues :

- `refined_scope_assessment=CONTEXTUALLY_STABLE_WITH_PRECONDITIONS`
- `usage_preconditions`
- `positive_usage_contexts`
- `negative_usage_contexts`
- `blocked_contexts`
- `truth_status=NOT_REEVALUATED_BY_A38`
- `revision_performed=false`
- `wrong_confirmations=0`

Run du 2026-06-19 :

- entrees lues : A33, A35, A37
- artefact explicitement non modifie : A33
- mecanique raffinee :
  `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- `source_scope_assessment=CONTEXTUALLY_STABLE`
- `refined_scope_assessment=CONTEXTUALLY_STABLE_WITH_PRECONDITIONS`
- `positive_usage_contexts=1`
- `negative_usage_contexts=1`
- `blocked_contexts=1`
- `effect_saturation_detected=true`
- `truth_status=NOT_REEVALUATED_BY_A38`
- `revision_performed=false`
- `wrong_confirmations=0`

Usage positif :

| Step | Contexte | Action | Signal | Patch avant | Patch apres | Lecture |
|---:|---|---|---:|---|---|---|
| 0 | `reset_exact` | `ACTION6` | 1 | `[[5,5,5],[3,5,3]]` | `[[5,5,5],[3,5,10]]` | patch non sature, usage utile |

Usage negatif :

| Step | Contexte A37 | Contexte bloque | Action | Signal | Patch avant | Patch apres | Cause probable |
|---:|---|---|---|---:|---|---|---|
| 2 | `after_ACTION3` | `after_ACTION3_live_after_ACTION6` | `ACTION6` | 0 | `[[5,5,5],[3,5,10]]` | `[[5,5,5],[3,5,10]]` | `target_patch_already_saturated` |

Preconditions d'usage extraites :

- `local_patch_available=true`
- `predicted_metric_signal_available=true`
- `selected_signal_expected=1`
- `target_patch_not_already_saturated=true`

Lecture :

- A38 introduit bien la notion de saturation d'effet : apres un premier
  `ACTION6`, le patch local cible contient deja `10`, donc rejouer `ACTION6`
  dans la trajectoire vivante ne produit plus le signal local attendu.
- Le signal negatif en rollout n'est pas une refutation de la mecanique A32/A33.
  Il raffine son applicabilite.
- Le scope A35 `CONTEXTUALLY_STABLE` devient
  `CONTEXTUALLY_STABLE_WITH_PRECONDITIONS`.
- La precondition centrale est :
  `target_patch_not_already_saturated=true`.

## Commandes de verification

Tests A38 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a38_rollout_aware_scope_refinement.py -q
```

Run A38 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a38.rollout_aware_scope_refinement --registry diagnostics\a33\confirmed_mechanics_registry.json --scope-map diagnostics\a35\confirmed_mechanic_scope_map.json --rollout diagnostics\a37\scope_conditioned_policy_rollout.json --out diagnostics\a38\rollout_aware_scope_refinement.json
```

Guard A38 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a38\*.py,diagnostics\a38\*.json,A38_MILESTONES.md -Pattern $patterns -SimpleMatch
```
