# A35 milestones - contextual scope mapper

Derniere mise a jour : 2026-06-19

A35 cartographie le scope d'usage d'une mecanique confirmee par A33. Il ne
confirme pas, ne refute pas, et ne modifie pas le registre A33. Il teste
seulement dans quels contextes courts la mecanique confirmee reste utile pour
l'action.

## Principe fixe

A35 ne dit pas : "la mecanique est vraie".

A35 dit seulement ou la mecanique deja confirmee par A32/A33 semble utile :

- `LOCAL_ONLY`
- `CONTEXTUALLY_STABLE`
- `PRECONDITION_DEPENDENT`
- `UNSTABLE_OR_NOT_USEFUL`

Le champ `truth_status` reste toujours :

- `NOT_REEVALUATED_BY_A35`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A35.1 - Contextual scope mapper | Fait | `theory/a35/confirmed_mechanic_scope_map.py`, `tests/test_a35_confirmed_mechanic_scope_map.py`, `diagnostics/a35/confirmed_mechanic_scope_map.json` | 4 contextes bp35 testes; scope `CONTEXTUALLY_STABLE`; aucun verdict de verite |

## A35.1 - Contextual scope mapper

Entrees :

- `diagnostics/a33/confirmed_mechanics_registry.json`
- `diagnostics/a34/confirmed_mechanic_usage_probe.json`

Sortie :

- `diagnostics/a35/confirmed_mechanic_scope_map.json`

Contrat :

- Lire uniquement les mecaniques confirmees deja presentes dans A33.
- Utiliser A34 comme reference d'usage, pas comme preuve supplementaire.
- Rejouer des contextes courts depuis `RESET`.
- Comparer une baseline neutre et l'action issue du registre A33.
- Mesurer la meme metrique : `local_patch_before_after`.
- Ne jamais produire de verdict de verite.

Contextes cibles :

- `reset_exact`
- `after_ACTION3`
- `after_ACTION4`
- `after_ACTION3_then_ACTION4`

Mesures par contexte :

- hypothese / mecanique
- action baseline neutre
- action issue du registre A33
- metrique
- `local_patch_before_after_observed`
- `useful_new_state`
- `functional_progress`
- `usage_contradiction`
- `score_or_level_unchanged_or_improved`
- `truth_status=NOT_REEVALUATED_BY_A35`

Run du 2026-06-19 :

- mecanique testee :
  `mechanic_prediction::bp35-0a0ad940::ACTION6::position_effect_candidate::local_patch_before_after`
- action issue du registre A33 : `ACTION6`
- baseline neutre : `ACTION3`
- metrique : `local_patch_before_after`
- contextes testes : 4
- `functional_progress_contexts=4`
- `useful_new_state_contexts=4`
- `usage_contradictions=0`
- `errors=0`
- `scope_assessment=CONTEXTUALLY_STABLE`
- `truth_status=NOT_REEVALUATED_BY_A35`
- `revision_performed=false`
- `wrong_confirmations=0`

Experiences documentees :

| Contexte | Baseline | Treatment | Baseline signal | Treatment signal | Support d'usage | Contradiction | Statut final |
|---|---|---|---:|---:|---|---|---|
| `reset_exact` | `ACTION3` | `ACTION6` | 0 | 1 | `functional_progress=true`, `useful_new_state=true` | false | `NOT_REEVALUATED_BY_A35` |
| `after_ACTION3` | `ACTION3` | `ACTION6` | 0 | 1 | `functional_progress=true`, `useful_new_state=true` | false | `NOT_REEVALUATED_BY_A35` |
| `after_ACTION4` | `ACTION3` | `ACTION6` | 0 | 1 | `functional_progress=true`, `useful_new_state=true` | false | `NOT_REEVALUATED_BY_A35` |
| `after_ACTION3_then_ACTION4` | `ACTION3` | `ACTION6` | 0 | 1 | `functional_progress=true`, `useful_new_state=true` | false | `NOT_REEVALUATED_BY_A35` |

Lecture :

- Sur ces quatre contextes courts bp35, la mecanique ACTION6 reste utile apres
  `RESET`, apres `ACTION3`, apres `ACTION4`, et apres `ACTION3` puis `ACTION4`.
- Le scope observe est donc plus large que le reset exact dans ce micro-domaine :
  `CONTEXTUALLY_STABLE`.
- Cette stabilite d'usage n'est pas une nouvelle confirmation de verite. La
  verite reste celle decidee par A32 ; A35 ne cartographie que l'applicabilite.
- Une future absence d'utilite dans un contexte plus long ne devrait pas refuter
  la mecanique ; elle reduirait le scope d'usage.

## Commandes de verification

Tests A35 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a35_confirmed_mechanic_scope_map.py -q
```

Run A35 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a35.confirmed_mechanic_scope_map --registry diagnostics\a33\confirmed_mechanics_registry.json --usage-probe diagnostics\a34\confirmed_mechanic_usage_probe.json --out diagnostics\a35\confirmed_mechanic_scope_map.json
```

Guard A35 :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1'
)
Select-String -Path theory\a35\*.py,diagnostics\a35\*.json,A35_MILESTONES.md -Pattern $patterns -SimpleMatch
```
