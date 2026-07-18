# A33 milestones - confirmed mechanic registry

Derniere mise a jour : 2026-07-18

A33 separe la memoire scientifique confirmee des files de decision. Il ne relit
pas M3. Il consomme uniquement les decisions A32.

## Principe fixe

A33 ne confirme rien lui-meme. Il enregistre seulement les decisions A32 dont :

- A33.1 : `decision=REVISION_ACCEPTED_AS_CONFIRMED` et
  `decision_record.status=confirmed` ;
- A33.2 :
  `decision=CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION`,
  `decision_record.status=confirmed` et un handoff A33 scope-locked explicite.

A33 exclut explicitement :

- support reutilise comme support independant
- support trace
- support prior
- candidats unresolved

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A33.1 - Confirmed Mechanic Registry | Fait | `theory/a33/confirmed_mechanics_registry.py`, `tests/test_a33_confirmed_mechanics_registry.py`, `diagnostics/a33/confirmed_mechanics_registry.json` | 1 mecanique confirmee locale issue de A32 |
| A33.2 - Scoped Unknown-game Registry | Fait - ACTION5 enregistree, ACTION6 exclue | `theory/a33/scoped_unknown_game_registry.py`, `tests/test_a33_scoped_unknown_game_registry.py`, `diagnostics/a33/scoped_unknown_game_registry.json` | Consomme uniquement le handoff confirme A32.5; verrouille jeu/candidat/contextes/mesure; importe support=4 sans nouveau verdict; conserve A33.1 intact |

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

## A33.2 - Scoped Unknown-game Registry

Objectif :

- Lire uniquement
  `diagnostics/a32/unknown_game_parameterized_control_revision_decisions.json`.
- Verifier que la revision A32.5 est complete et que chaque handoff :
  - correspond a une decision
    `CONFIRM_SCOPE_LIMITED_AFTER_PARAMETERIZED_CONTROL_REVISION` ;
  - possede un `decision_record.status=confirmed` ;
  - a un support positif, zero contradiction et des identifiants d'experience
    complets ;
  - conserve le jeu, le candidat, l'action, la mesure, les contextes et les
    controles pre-enregistres ;
  - interdit toute generalisation au-dela du scope A32.5.
- Enregistrer la confirmation A32.5 sans reevaluer sa verite et sans produire
  une nouvelle confirmation A33.
- Exclure explicitement les records unresolved et non identifiables.
- Ne pas modifier le registre A33.1 historique ni ses consommateurs A34-A39,
  dont le contrat ne porte pas les contextes replay et arguments unknown-game.

Artefacts :

- `theory/a33/scoped_unknown_game_registry.py`
- export dans `theory/a33/__init__.py`
- `tests/test_a33_scoped_unknown_game_registry.py`
- `diagnostics/a33/scoped_unknown_game_registry.json`

Run du 2026-07-18 :

- `source_candidates_consumed=2`
- `a32_5_handoff_candidates_consumed=1`
- `scoped_confirmed_mechanics_registered=1`
- `unresolved_candidates_excluded=1`
- `non_identifiable_candidates_excluded=1`
- `confirmed_support_imported_from_a32_5=4`
- `experiments_spent_total=4`
- `registered_contexts=4`
- `registered_parameterized_control_variants=2`
- `scope_locked_entries=1`
- `a33_truth_reevaluations=0`
- `a33_confirmations_performed=0`
- `legacy_a33_1_registry_mutated=false`
- `scope_generalization_performed=false`
- `wrong_confirmations=0`
- `outcome_status=A33_SCOPED_UNKNOWN_GAME_REGISTRY_ENTRY_ADDED`

Mecanique inscrite :

- key :
  `mechanic_prediction::sb26-7fbdac44::ACTION5::local_patch_change_candidate::local_patch_before_after::args=null`
- `game_id=sb26-7fbdac44`
- `action=ACTION5`
- `action_args=null`
- `mechanic_family=local_patch_change_candidate`
- `predicted_metric=local_patch_before_after`
- `confirmed_support=4`
- `contradictions=0`
- `experiments_spent=4`
- `budgets=[50,300]`
- 4 contextes replay-exacts verrouilles
- controles parametres : `ACTION6 {"x":21,"y":28}` et
  `ACTION6 {"x":39,"y":28}`
- `known_scope=game_candidate_contexts_measurement`
- tous les marqueurs `not_generalized_*` restent vrais.

Candidat exclu :

- `ACTION6 {"x":26,"y":57}`
- `decision_record.status=unresolved`
- `support=0`
- `exclusion_reason=NON_IDENTIFIABLE_UNRESOLVED_NOT_REGISTRY_ELIGIBLE`
- `registered=false`
- `counted_as_refutation=false`

Lecture :

- A33.2 ne reconfirme pas ACTION5 ; il transporte fidelement le verdict A32.5
  vers une memoire confirmee scopee.
- Le support `4` garde pour origine les `decision_record` A32.5. Aucun evenement
  SAGE candidate-only n'est recompte par A33.
- La separation avec A33.1 empeche les anciens usages A34-A39 d'appliquer
  ACTION5 hors des quatre contextes observes.
- La boucle SAGE.5j -> A32.5 -> A33.2 est maintenant fermee sans enregistrer le
  candidat ACTION6 non identifiable.
- SAGE.6 consomme ensuite A33.2 en lecture seule comme garde de quarantaine :
  l'entree ACTION5 reste sur `sb26` et n'est pas appliquee au second jeu
  `wa30-ee6fef47`.

## Commandes de verification

Tests A33 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a33_confirmed_mechanics_registry.py tests\test_a33_scoped_unknown_game_registry.py -q
```

Generation registre :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a33.confirmed_mechanics_registry --decisions diagnostics\a32\a15_revision_decisions.json --out diagnostics\a33\confirmed_mechanics_registry.json
```

Generation registre unknown-game scope :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a33.scoped_unknown_game_registry --decisions diagnostics\a32\unknown_game_parameterized_control_revision_decisions.json --out diagnostics\a33\scoped_unknown_game_registry.json
```
