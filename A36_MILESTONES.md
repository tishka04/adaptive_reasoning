# A36 milestones - scope-conditioned action policy

Derniere mise a jour : 2026-06-19

A36 est le premier "cerveau moteur" minimal : il utilise la memoire
scientifique A33 et le scope A35 pour choisir une action en contexte live. Il
ne relit pas M3, A32, ni A34, et il ne refait pas de revision scientifique.

## Principe fixe

A36 ne dit pas : "la mecanique est vraie".

A36 dit seulement :

- une mecanique confirmee existe dans A33 ;
- son scope A35 est utilisable dans le contexte courant ;
- la policy peut prioriser son action ;
- l'action choisie produit ou non un progres fonctionnel.

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A36`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A36.1 - Scope-conditioned policy probe | Fait | `theory/a36/scope_conditioned_policy_probe.py`, `tests/test_a36_scope_conditioned_policy_probe.py`, `diagnostics/a36/scope_conditioned_policy_probe.json` | ACTION6 selectionnee depuis A33+A35; progres fonctionnel observe; aucun verdict de verite |

## A36.1 - Scope-conditioned policy probe

Entrees :

- `diagnostics/a33/confirmed_mechanics_registry.json`
- `diagnostics/a35/confirmed_mechanic_scope_map.json`

Sortie :

- `diagnostics/a36/scope_conditioned_policy_probe.json`

Contrat :

- Si une mecanique confirmee existe dans A33.
- Et si son scope A35 est `CONTEXTUALLY_STABLE`.
- Et si le contexte live est couvert ou voisin du scope.
- Alors prioriser l'action de cette mecanique.
- Sinon utiliser une baseline neutre / exploration.

Mesures :

- `policy_selected_action`
- `selected_from_confirmed_mechanic`
- `scope_used`
- `context_match`
- `functional_progress`
- `useful_new_state`
- `usage_contradiction`
- `wrong_confirmations=0`
- `truth_status=NOT_REEVALUATED_BY_A36`

Run du 2026-06-19 :

- entrees lues : A33 et A35 uniquement
- artefacts non lus pour la decision : M3, A32, A34
- mecanique candidate policy :
  `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- contexte live : `after_ACTION3_then_ACTION4`
- `scope_used=CONTEXTUALLY_STABLE`
- `context_match=true`
- `context_match_reason=covered_context_exact`
- `policy_selected_action=ACTION6`
- `fallback_action=ACTION3`
- `selected_from_confirmed_mechanic=true`
- `selected_signal=1`
- `fallback_signal=0`
- `functional_progress=true`
- `useful_new_state=true`
- `usage_contradiction=false`
- `truth_status=NOT_REEVALUATED_BY_A36`
- `revision_performed=false`
- `wrong_confirmations=0`

Experience documentee :

| Contexte live | Scope utilise | Action policy | Fallback | Signal policy | Signal fallback | Progres | Contradiction | Statut final |
|---|---|---|---|---:|---:|---|---|---|
| `after_ACTION3_then_ACTION4` | `CONTEXTUALLY_STABLE` | `ACTION6` | `ACTION3` | 1 | 0 | `functional_progress=true`, `useful_new_state=true` | false | `NOT_REEVALUATED_BY_A36` |

Lecture :

- A36 agit avec la memoire scientifique sans refaire la science : la selection
  vient du registre A33 et du scope A35.
- Dans le contexte couvert `after_ACTION3_then_ACTION4`, la policy priorise
  `ACTION6` et observe le signal local attendu.
- Cette utilite n'est pas une confirmation supplementaire. La verite reste le
  verdict A32 ; A36 mesure seulement l'usage moteur de cette memoire.
- Un futur echec d'usage A36 devrait etre traite comme un signal de politique ou
  de scope, pas comme une refutation de la mecanique.

## Commandes de verification

Tests A36 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a36_scope_conditioned_policy_probe.py -q
```

Run A36 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a36.scope_conditioned_policy_probe --registry diagnostics\a33\confirmed_mechanics_registry.json --scope-map diagnostics\a35\confirmed_mechanic_scope_map.json --context-sequence ACTION3 ACTION4 --out diagnostics\a36\scope_conditioned_policy_probe.json
```

Guard A36 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a36\*.py,diagnostics\a36\*.json,A36_MILESTONES.md -Pattern $patterns -SimpleMatch
```
