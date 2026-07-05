# A40 milestones - frontier handoff to scientific loop

Derniere mise a jour : 2026-06-19

A40 transforme les impasses de policy A39 en requetes scientifiques explicites
pour M1/M3. Il ne teste pas, ne confirme pas, ne refute pas, et ne modifie pas
les artefacts de memoire.

## Principe fixe

A40 ne dit pas : "la mecanique est vraie ou fausse".

A40 dit seulement :

- la policy a rencontre une frontiere d'usage ;
- cette frontiere doit redevenir une question scientifique ;
- M1/M3 peuvent generer ou tester de nouvelles hypotheses depuis cet etat.

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A40`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A40.1 - Frontier handoff requests | Fait | `theory/a40/frontier_handoff_requests.py`, `tests/test_a40_frontier_handoff_requests.py`, `diagnostics/a40/frontier_handoff_requests.json` | 2 frontieres creees; ready for M1/M3; aucun verdict de verite |

## A40.1 - Frontier handoff requests

Entree :

- `diagnostics/a39/precondition_aware_policy_rollout.json`

Sortie :

- `diagnostics/a40/frontier_handoff_requests.json`

Contrat :

- Detecter les competences bloquees par precondition echouee.
- Detecter les contextes non couverts par le scope.
- Detecter les fallbacks sans progres.
- Produire des `frontier_requests` consommables par M1/M3.
- Ne jamais executer l'environnement.
- Ne jamais reviser la verite scientifique.

KPIs :

- `frontier_requests_created`
- `blocked_skill_frontiers`
- `uncovered_context_frontiers`
- `fallback_no_progress_frontiers`
- `ready_for_m1_or_m3`
- `wrong_confirmations=0`
- `truth_status=NOT_REEVALUATED_BY_A40`

Run du 2026-06-19 :

- entree lue : A39 uniquement
- environnement execute : non
- revision scientifique : non
- artefacts non modifies : A33, A35, A38, A39
- `frontier_requests_created=2`
- `blocked_skill_frontiers=1`
- `uncovered_context_frontiers=1`
- `fallback_no_progress_frontiers=2`
- `cycle_or_dead_end_frontiers=0`
- `ready_for_m1_or_m3=true`
- `truth_status=NOT_REEVALUATED_BY_A40`
- `revision_performed=false`
- `wrong_confirmations=0`

Requetes creees :

| Source step | Frontier | Raison | Skill bloquee | Precondition echouee | Action recommandee |
|---:|---|---|---|---|---|
| 1 | `after_ACTION6` | `context_not_covered_by_scope` |  |  | `generate_new_candidate_mechanics_from_current_state` |
| 2 | `after_ACTION3_live_after_ACTION6` | `confirmed_skill_blocked_by_failed_precondition` | `ACTION6` | `target_patch_not_already_saturated=true` | `generate_new_candidate_mechanics_from_current_state` |

Lecture :

- A39 apprend a ne pas faire une mauvaise action.
- A40 transforme ce "je ne sais pas quoi faire ensuite" en requete
  scientifique explicite.
- La frontiere `after_ACTION6` signale un contexte non couvert par le scope.
- La frontiere `after_ACTION3_live_after_ACTION6` signale une competence
  confirmee inhibee par precondition dynamique.
- Le handoff est une file d'apprentissage, pas une preuve.

## Commandes de verification

Tests A40 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a40_frontier_handoff_requests.py -q
```

Run A40 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a40.frontier_handoff_requests --policy-rollout diagnostics\a39\precondition_aware_policy_rollout.json --out diagnostics\a40\frontier_handoff_requests.json
```

Guard A40 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a40\*.py,diagnostics\a40\*.json,A40_MILESTONES.md -Pattern $patterns -SimpleMatch
```
