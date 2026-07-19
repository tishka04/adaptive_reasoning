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
| A34.2 - Control-dependent relational usage probe | Fait | `theory/a34/control_dependent_relational_usage_probe.py`, `tests/test_a34_control_dependent_relational_usage_probe.py`, `diagnostics/a34/control_dependent_relational_usage_probe.json` | ACTION2 priorisee dans les 3 contextes A33.3; gain local +32 face a ACTION1, egalite avec ACTION3; aucun niveau ni win |

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

## A34.2 - Control-dependent relational usage probe

Entrees :

- `diagnostics/a33/control_dependent_relational_registry.json` pour la relation
  et le choix d'action ;
- `diagnostics/sage/sage6f_second_unknown_game_control_dependence_consolidation.json`
  pour relier les trois hashes A33.3 a leur provenance ;
- `diagnostics/sage/sage6a_switch_attribution_mini_frontier.json` uniquement
  pour reconstruire les prefixes live exacts.

Sortie :

- `diagnostics/a34/control_dependent_relational_usage_probe.json`

Contrat :

- Baseline sans memoire : `ACTION1`, le comparateur de plus faible effet
  enregistre par A33.3.
- Treatment avec memoire : `ACTION2`, uniquement quand le jeu, la metrique et
  le hash de contexte correspondent exactement au scope A33.3.
- Audit de limite : `ACTION3`, le controle equivalent enregistre.
- Les trois bras sont executes depuis le meme prefixe live rejoue et verifie.
- Les sources SAGE servent seulement a reconstruire le contexte ; elles ne
  choisissent jamais l'action.
- A34.2 mesure l'utilite sans reconfirmer la relation ni recompter son support.

Run du 2026-07-19 :

- `registered_relations_probed=1`
- `exact_contexts_probed=3`
- budgets `[50,150,300]`, steps `[48,132,24]`
- `action_choices_changed=3`
- baseline `ACTION1`, signal `[1,1,0]`
- treatment A33.3 `ACTION2`, signal `[33,33,32]`
- audit equivalent `ACTION3`, signal `[33,33,32]`
- `registry_gain_over_baseline=[32,32,32]`
- `registry_gain_over_equivalent=[0,0,0]`
- `contextual_relational_utility_events=3`
- `functional_local_progress_events=3`
- `registry_levels_completed_delta_total=0`
- `registry_levels_completed_max=0`
- `registry_wins=0`
- `registry_win_rate=0`
- `level_or_win_progress_demonstrated=false`
- `support_counted=0`
- `wrong_confirmations=0`
- `outcome_status=A34_CONTROL_DEPENDENT_RELATION_CONTEXTUALLY_USEFUL`

Lecture :

- La memoire A33.3 evite effectivement `ACTION1` dans chacun des trois
  contextes autorises et reproduit le gain relationnel local `+32`.
- `ACTION3` reste exactement equivalent a `ACTION2`. A34.2 ne peut donc pas
  attribuer une utilite autonome a ACTION2 ni pretendre que la relation impose
  un choix unique.
- Le changement de choix produit un signal fonctionnel local, mais aucun niveau
  ni aucune victoire. C'est une preuve d'utilite decisionnelle intermediaire,
  pas encore un progres ARC-AGI-3.

## Commandes de verification

Tests A34 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a34_confirmed_mechanic_usage_probe.py -q
```

Run A34 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a34.confirmed_mechanic_usage_probe --registry diagnostics\a33\confirmed_mechanics_registry.json --out diagnostics\a34\confirmed_mechanic_usage_probe.json
```

Run A34.2 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a34.control_dependent_relational_usage_probe --registry diagnostics\a33\control_dependent_relational_registry.json --source-sage6f diagnostics\sage\sage6f_second_unknown_game_control_dependence_consolidation.json --source-sage6a diagnostics\sage\sage6a_switch_attribution_mini_frontier.json --out diagnostics\a34\control_dependent_relational_usage_probe.json
```
