# P2 Milestones - Policy Frontier Extraction

Derniere mise a jour : 2026-06-21

P2 transforme les saturations runtime P1 en FrontierRecord exploitables par la boucle scientifique. P2 ne confirme rien, ne refute rien, et ne force jamais une frontier quand P1 n'en a pas produit une dans un rollout reel.

## Principes fixes

- P2 peut lire les artefacts P1 explicitement candidate-only.
- P2 ne lit pas A33.
- P2 n'ecrit pas A40/M2/M3/A32/A33.
- P2 ne confirme pas, ne refute pas, et ne modifie aucun support scientifique.
- Les fixtures synthetiques peuvent valider un schema, mais ne sont jamais des handoffs scientifiques.
- Tous les artefacts P2 gardent `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`, `wrong_confirmations=0`.

## Etat

| Milestone | Statut | Artefacts | Resultat |
| --- | --- | --- | --- |
| P2.1 - Policy FrontierRecord schema | Fait | `theory/p2/policy_frontier_records.py`, `tests/test_p2_policy_frontier_records.py`, `diagnostics/p2/bp35_policy_frontier_records.json` | P1.11 ne contient aucune frontier reelle, donc aucun handoff P2. La fixture synthetique valide le schema sans devenir scientifique. |
| P2.2 - Policy FrontierRecord handoff validator | Fait | `theory/p2/policy_frontier_handoff_validator.py`, `tests/test_p2_policy_frontier_handoff_validator.py`, `diagnostics/p2/bp35_policy_frontier_handoff_validation.json` | Le record synthetique P2.1 est rejete proprement. Aucun write A40/M2/M3. |
| P2.3 - Long-budget saturation probe | Fait | `theory/p2/long_budget_saturation_probe.py`, `tests/test_p2_long_budget_saturation_probe.py`, `diagnostics/p2/bp35_long_budget_saturation_probe.json` | 18/18 runs sans saturation refresh. Aucun handoff P2.4 pret. Le budget consomme ne compte jamais comme saturation. |
| P2.4-terminal - Terminal outcome frontier classifier | Fait | `theory/p2/terminal_outcome_frontier.py`, `tests/test_p2_terminal_outcome_frontier.py`, `diagnostics/p2/bp35_terminal_outcome_frontier.json` | 15/18 runs classifies comme `OBJECTIVE_ALIGNMENT_FRONTIER`. Pas de handoff saturation P2.4. |
| P2.5 - Objective frontier handoff validator | Fait | `theory/p2/objective_frontier_handoff_validator.py`, `tests/test_p2_objective_frontier_handoff_validator.py`, `diagnostics/p2/bp35_objective_frontier_handoff_validation.json` | 1 frontier objectif acceptee pour review no-write. 0 handoff saturation, 0 write A40/M2/M3/A32. |
| P2.6 - Objective frontier handoff schema | Fait | `theory/p2/objective_frontier_handoff_schema.py`, `tests/test_p2_objective_frontier_handoff_schema.py`, `diagnostics/p2/bp35_objective_frontier_handoff_requests.json` | 1 requete `OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST` cible `M2_OR_M3`, sans write aval et sans support. |
| P2.G1 - Objective-conversion frontier records | Fait | `theory/p2/objective_conversion_frontier_records.py`, `tests/test_p2_objective_conversion_frontier_records.py`, `diagnostics/p2/objective_conversion_frontier_records.json` | 1 `OBJECTIVE_CONVERSION_FRONTIER` issue de P3.G1 safe-but-passive. Review objective-conversion prete, mais pas de handoff direct M2/M3. |
| P2.G2 - Objective-conversion frontier validator | Fait | `theory/p2/objective_conversion_frontier_validator.py`, `tests/test_p2_objective_conversion_frontier_validator.py`, `diagnostics/p2/objective_conversion_frontier_validation.json` | La frontier P2.G1 est acceptee pour review no-write. 0 write A40/M2/M3/A32/A33 et `ready_for_m2_or_m3=false`. |
| P2.G3 - Objective-conversion handoff schema | Fait | `theory/p2/objective_conversion_handoff_schema.py`, `tests/test_p2_objective_conversion_handoff_schema.py`, `diagnostics/p2/objective_conversion_handoff_requests.json` | 1 requete `OBJECTIVE_CONVERSION_FRONTIER_REQUEST` cible `M2_OR_M3`, modules `M2.G1`/`M3.G1`, sans write aval direct. |
| P2.G4 - Risk-aware post-stop policy frontier record | Fait | `theory/p2/risk_aware_post_stop_frontier_records.py`, `tests/test_p2_risk_aware_post_stop_frontier_records.py`, `diagnostics/p2/risk_aware_post_stop_frontier_records.json` | 1 `RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER` issue de P3.G4. Safety terminale, utilite OOS et selection risk-aware observees, mais completion objectif absente. Pas de handoff direct M2/M3. |
| P2.G5 - Risk-aware objective frontier validator/schema | Fait | `theory/p2/risk_aware_objective_frontier_handoff_schema.py`, `tests/test_p2_risk_aware_objective_frontier_handoff_schema.py`, `diagnostics/p2/risk_aware_objective_frontier_handoff_requests.json` | La frontier P2.G4 est acceptee pour review no-write et transformee en 1 requete `RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST` vers `M2.G2`, sans write aval direct. |

## P2.1 - Policy FrontierRecord schema

Objectif :

Definir le format universel `PolicyFrontierRecord` depuis le trigger runtime P1.11.

Entree :

- `diagnostics/p1/bp35_movement_policy_frontier_trigger.json`

Sortie :

- `diagnostics/p2/bp35_policy_frontier_records.json`

Contrat :

- Cas reel P1.11 sans frontier : `real_frontier_records=[]`.
- Cas fixture synthetique : record conserve pour validation de schema uniquement.
- Un record synthetique ne peut jamais etre `ready_for_m2_or_m3=true`.
- Aucun handoff scientifique n'est produit tant qu'il n'existe pas de frontier reelle.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.policy_frontier_records `
  --p1-frontier-trigger diagnostics\p1\bp35_movement_policy_frontier_trigger.json `
  --out diagnostics\p2\bp35_policy_frontier_records.json
```

Resume du run :

- real_frontier_records = 0
- synthetic_frontier_records = 1
- real_handoffs_produced = 0
- ready_for_m2_or_m3_records = 0
- synthetic_records_for_schema_validation_only = 1
- synthetic_records_counted_as_handoff = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Record synthetique schema-only :

```json
{
  "frontier_id": "p1::bp35::conditional_movement_refresh::exhausted_refresh::synthetic_fixture",
  "game_id": "bp35-0a0ad940",
  "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH",
  "exhausted_policy": "conditional_movement_refresh",
  "refresh_candidates": ["ACTION3", "ACTION4", "ACTION1", "ACTION2"],
  "ready_for_m2_or_m3": false,
  "synthetic_fixture_not_for_scientific_handoff": true,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY"
}
```

Lecture :

P2.1 ne force pas une frontier. Il constate que P1.11 n'a pas produit de saturation reelle sur bp35, donc aucun handoff vers M2/M3 n'est autorise. La fixture synthetique prouve seulement que le format `PolicyFrontierRecord` est valide pour le jour ou une vraie saturation runtime apparaitra.

Suite naturelle :

- P2.3 - Long-budget rollout pour provoquer une saturation reelle, si elle existe.
- P2.4 - Handoff vers M2/M3 seulement si P2 observe une vraie frontier.

## P2.2 - Policy FrontierRecord handoff validator

Objectif :

Verifier quels `PolicyFrontierRecord` peuvent etre transmis a A40/M2/M3, sans effectuer encore aucun write.

Entree :

- `diagnostics/p2/bp35_policy_frontier_records.json`

Sortie :

- `diagnostics/p2/bp35_policy_frontier_handoff_validation.json`

Contrat :

- Les records synthetiques sont toujours rejetes.
- Les records non `ready_for_m2_or_m3` sont rejetes.
- Le validateur ne produit aucun fichier A40/M2/M3.
- Un handoff accepte doit etre reel, candidate-only, `support=0`, et sans flags de verdict.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.policy_frontier_handoff_validator `
  --policy-frontier-records diagnostics\p2\bp35_policy_frontier_records.json `
  --out diagnostics\p2\bp35_policy_frontier_handoff_validation.json
```

Resume du run :

- records_seen = 1
- real_records_seen = 0
- synthetic_records_seen = 1
- handoffs_accepted = 0
- handoffs_rejected = 1
- rejection_reasons = `NOT_READY_FOR_M2_OR_M3`, `SYNTHETIC_FIXTURE_NOT_FOR_SCIENTIFIC_HANDOFF`
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`

Lecture :

P2.2 verrouille le garde-fou attendu : la fixture synthetique issue de P1.11/P2.1 valide le schema, mais elle ne peut pas reveiller A40/M2/M3. Dans l'etat actuel, il n'y a donc aucun handoff scientifique.

Suite naturelle :

- P2.3 - Long-budget rollout pour chercher une saturation runtime reelle.
- P2.4 - Handoff A40/M2/M3 seulement si P2.3 produit un record reel accepte.

## P2.3 - Long-budget saturation probe

Objectif :

Chercher si la policy `conditional_movement_refresh` finit par saturer sur des budgets longs, sans confondre fin de budget et frontier scientifique.

Entree :

- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Sortie :

- `diagnostics/p2/bp35_long_budget_saturation_probe.json`

Contrat :

- Tester uniquement `conditional_movement_refresh`.
- Budgets par defaut : 48, 64, 96, 128, 192, 256.
- Seeds deterministes par defaut : 0, 1, 2.
- Une vraie frontier exige :
  - aucun `ACTION6` effectif disponible,
  - un mouvement refresh tente,
  - une observation post-refresh sans nouvel `ACTION6` utile.
- `budget_exhausted` ne compte jamais comme saturation.
- Aucun write A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.long_budget_saturation_probe `
  --scope-consolidation diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p2\bp35_long_budget_saturation_probe.json `
  --budgets 48 64 96 128 192 256 `
  --tie-break-seeds 0 1 2
```

Lecture attendue :

- `NO_SATURATION_ACTION6_REMAINS_PRODUCTIVE` : pas de frontier, la policy continue.
- `SATURATION_REFRESH_UNLOCKED_ACTION6` : saturation locale, mais un mouvement refresh rouvre ACTION6.
- `TRUE_FRONTIER_AFTER_FAILED_MOVEMENT_REFRESH` : vraie frontier candidate pour P2.4.
- `SATURATION_REFRESH_ATTEMPTED_POST_REFRESH_UNOBSERVED` : refresh tente, mais pas assez d'observation apres refresh pour conclure.

Resume du run :

- budget_runs = 18
- budgets_tested = 48, 64, 96, 128, 192, 256
- tie_break_seeds_tested = 0, 1, 2
- outcome_counts = `NO_SATURATION_ACTION6_REMAINS_PRODUCTIVE`: 18
- no_saturation_runs = 18
- movement_refresh_unlock_runs = 0
- true_frontier_runs = 0
- post_refresh_unobserved_runs = 0
- budget_exhausted_runs = 6
- budget_exhausted_counted_as_saturation = false
- real_frontier_ready_for_p2_4 = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Lecture :

P2.3 ne trouve pas de vraie frontier runtime sur les budgets longs. Les runs 48 consomment leur budget, mais cela n'est pas compte comme saturation. Les budgets 64+ s'arretent autour de 65 pas avec `GAME_OVER`, 64 `ACTION6` utiles, 6 args uniques et 59 repetitions, sans aucun trigger `conditional_movement_refresh_after_soft_stale_exhausted`.

La conclusion propre est donc : la policy `conditional_movement_refresh` n'a pas besoin de mouvement refresh sur ce protocole bp35 long-budget, et P2 ne doit pas reveiller A40/M2/M3. Si une nouvelle frontier est necessaire, elle devra venir d'un autre critere runtime que "budget long consomme", par exemple une limite d'objectif, un plateau de score externe, ou un autre contexte de jeu.

Suite naturelle :

- P2.4 ne doit pas etre lance sur ce run, car `real_frontier_ready_for_p2_4=false`.
- Prochaine branche possible : definir un critere d'objectif/terminalite plus riche si `GAME_OVER` sans niveau complete doit devenir une nouvelle question scientifique.

## P2.4-terminal - Terminal outcome frontier classifier

Objectif :

Classifier les cas ou la policy reste localement productive mais echoue globalement en terminalite. Cette branche est separee du handoff de saturation P2.4.

Entree :

- `diagnostics/p2/bp35_long_budget_saturation_probe.json`

Sortie :

- `diagnostics/p2/bp35_terminal_outcome_frontier.json`

Critere minimal :

- `final_game_state = GAME_OVER`
- `final_levels_completed = 0`
- `useful_action6_steps` eleve
- `conditional_movement_refresh_triggers = 0`
- `true_saturation_frontier = false`

Interpretation :

`LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED`

Contrat :

- Ne pas produire de handoff saturation P2.4.
- Ne pas ecrire A40/M2/M3/A32/A33.
- Ne pas confirmer ni refuter.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.terminal_outcome_frontier `
  --long-budget-saturation-probe diagnostics\p2\bp35_long_budget_saturation_probe.json `
  --out diagnostics\p2\bp35_terminal_outcome_frontier.json
```

Resume du run :

- runs_seen = 18
- final_game_state_counts = `GAME_OVER`: 15, `NOT_FINISHED`: 3
- terminal_objective_failure_runs = 15
- productive_local_affordance_terminal_runs = 15
- objective_alignment_frontier_runs = 15
- terminal_budgets = 64, 96, 128, 192, 256
- movement_refresh_triggers_total = 0
- true_saturation_frontier_runs = 0
- ready_for_p2_4_saturation_handoff = false
- ready_for_objective_frontier_review = true
- frontier_type = `OBJECTIVE_ALIGNMENT_FRONTIER`
- frontier_reason = `LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED`
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Lecture :

P2.4-terminal confirme que la limite observee par P2.3 n'est pas une saturation d'affordance. `ACTION6` reste productif localement, aucun mouvement refresh n'est appele, mais les budgets 64+ finissent en `GAME_OVER` avec 0 niveau complete. La frontier correcte est donc une question d'alignement objectif : quand une affordance locale utile cesse-t-elle de servir le but global ?

Suite naturelle :

- Ne pas lancer P2.4 saturation-handoff sur cet artefact.
- Definir une branche objective-frontier review/handoff separee si l'on veut demander a M2/M3 des hypotheses sur terminalite, stop-switch, ou sous-but global.

## P2.5 - Objective frontier handoff validator

Objectif :

Valider la recevabilite de la frontier d'alignement objectif P2.4-terminal, sans ecrire encore A40/M2/M3/A32/A33.

Entree :

- `diagnostics/p2/bp35_terminal_outcome_frontier.json`

Sortie :

- `diagnostics/p2/bp35_objective_frontier_handoff_validation.json`

Contrat :

- Accepter uniquement `frontier_type=OBJECTIVE_ALIGNMENT_FRONTIER`.
- Accepter uniquement `frontier_reason=LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED`.
- Exiger `ready_for_objective_frontier_review=true`.
- Refuser tout `ready_for_p2_4_saturation_handoff=true`.
- Refuser tout `ready_for_m2_or_m3=true` a ce stade.
- Aucun write A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.objective_frontier_handoff_validator `
  --terminal-outcome-frontier diagnostics\p2\bp35_terminal_outcome_frontier.json `
  --out diagnostics\p2\bp35_objective_frontier_handoff_validation.json
```

Resume du run :

- objective_frontiers_seen = 1
- objective_reviews_accepted = 1
- objective_reviews_rejected = 0
- saturation_handoffs_accepted = 0
- ready_for_p2_4_saturation_handoff = false
- ready_for_objective_frontier_review = true
- objective_review_no_write = true
- rejection_reasons = []
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Lecture :

P2.5 valide que la frontier d'objectif est recevable pour une review objective-frontier, mais ne la transforme pas encore en requete A40/M2/M3. La branche saturation reste fermee (`ready_for_p2_4_saturation_handoff=false`) et aucun artefact scientifique aval n'est mute.

Suite naturelle :

- P2.6 - Objective frontier handoff schema vers M2/M3, si l'on decide d'ouvrir explicitement cette branche.
- Ou A32/P3 objective review pour definir les questions stop-switch avant de reveiller M2/M3.

## P2.6 - Objective frontier handoff schema

Objectif :

Transformer la review P2.5 acceptee en requete objectif candidate-only pour la branche M2/M3, sans resoudre la frontier et sans ecrire dans M2/M3.

Entree :

- `diagnostics/p2/bp35_objective_frontier_handoff_validation.json`

Sortie :

- `diagnostics/p2/bp35_objective_frontier_handoff_requests.json`

Contrat :

- Produire `handoff_type=OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST`.
- Produire `target=M2_OR_M3`.
- Porter les questions stop/switch :
  - quand stopper `ACTION6` malgre effet local utile ?
  - quels signaux annoncent `GAME_OVER` ?
  - quel sous-but remplace l'exploitation repetee de `ACTION6` ?
  - quelle condition stop/switch est testable par M3 ?
- Refuser toute saturation handoff.
- Refuser `a33_ready=true`.
- Ne pas ecrire A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.objective_frontier_handoff_schema `
  --objective-frontier-validation diagnostics\p2\bp35_objective_frontier_handoff_validation.json `
  --out diagnostics\p2\bp35_objective_frontier_handoff_requests.json
```

Resume du run :

- source_objective_reviews_accepted = 1
- objective_frontier_requests = 1
- handoff_type = `OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST`
- target = `M2_OR_M3`
- target_modules = `M2.O1`, `M3.O1`
- ready_for_m2_or_m3_objective_branch = true
- ready_for_saturation_handoff = false
- saturation_handoff_requests = 0
- a33_ready = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Questions portees :

- Quand stopper `ACTION6` malgre un effet local utile ?
- Quels signaux predisent `GAME_OVER` pendant l'exploitation repetee de `ACTION6` ?
- Quel sous-but remplace l'exploitation repetee de `ACTION6` patch-similar ?
- Quelle condition stop/switch peut etre testee par M3 sans confirmation ?

Lecture :

P2.6 ouvre explicitement la branche objectif vers M2/M3, mais uniquement sous forme de requete candidate-only stockee dans `diagnostics/p2/`. Aucun artefact M2/M3/A32/A33 n'est modifie. La prochaine etape naturelle est `M2.O1`, qui devra transformer cette frontier objectif en hypotheses stop/switch falsifiables, et non en competence confirmee.

## P2.G1 - Objective-conversion frontier records

Objectif :

Transformer le resultat P3.G1 `POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY` en frontier objectif structuree, sans ecrire dans A40/M2/M3/A32/A33.

Entree :

- `diagnostics/p3/objective_aware_abstract_policy_utility_consolidation.json`

Sortie :

- `diagnostics/p2/objective_conversion_frontier_records.json`

Contrat :

- Declencher uniquement si P3.G1 reduit le terminal-rate mais ne produit aucun signal de completion objectif.
- Produire `frontier_type=OBJECTIVE_CONVERSION_FRONTIER`.
- Produire `frontier_reason=TERMINAL_SAFE_BUT_PASSIVE`.
- Marquer `blocked_capability=objective_conversion_after_safe_stop`.
- Garder `ready_for_objective_conversion_review=true`.
- Garder `ready_for_m2_or_m3=false` : P2.G1 ne reveille pas directement M2/M3.
- Ne jamais compter le resultat policy comme verdict scientifique.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.objective_conversion_frontier_records `
  --p3-objective-aware-consolidation diagnostics\p3\objective_aware_abstract_policy_utility_consolidation.json `
  --out diagnostics\p2\objective_conversion_frontier_records.json
```

Resume du run :

- source_policy_utility_status = `POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY`
- frontier_records = 1
- frontier_type = `OBJECTIVE_CONVERSION_FRONTIER`
- frontier_reason = `TERMINAL_SAFE_BUT_PASSIVE`
- terminal_safe_but_passive_frontiers = 1
- blocked_capability = `objective_conversion_after_safe_stop`
- ready_for_objective_conversion_review = true
- ready_for_m2_or_m3 = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Questions portees :

- Apres un stop/evitement terminal-safe, quelle action convertit le progres relationnel en completion objectif ?
- Quelle relation cible doit remplacer `distance_decreases` quand la completion reste fausse ?
- Une sequence post-safe-stop `ACTION3`, `ACTION4`, ou `ACTION6` cree-t-elle du progres objectif sans reentrer en risque terminal ?
- Quel signal d'objective-readiness manque au modele symbolique abstrait ?

Lecture :

P2.G1 isole le nouveau blocage : P3.G1 sait rendre la policy terminal-safe, mais elle devient passive et ne complete pas bp35. La frontier correcte n'est donc plus "eviter GAME_OVER", mais "convertir un etat safe en progres objectif". P2.G1 produit un dossier review-ready pour cette question, tout en refusant le handoff direct vers M2/M3 tant qu'un validateur ou schema aval dedie n'a pas ete ajoute.

Suite naturelle :

- P2.G2 - Objective-conversion frontier validator, no-write.
- Puis P2.G3 - Objective-conversion handoff schema, sans write M2/M3.

## P2.G2 - Objective-conversion frontier validator

Objectif :

Valider la recevabilite de la frontier P2.G1 pour une review objective-conversion, sans produire encore de requete M2/M3.

Entree :

- `diagnostics/p2/objective_conversion_frontier_records.json`

Sortie :

- `diagnostics/p2/objective_conversion_frontier_validation.json`

Contrat :

- Accepter uniquement `frontier_type=OBJECTIVE_CONVERSION_FRONTIER`.
- Accepter uniquement `frontier_reason=TERMINAL_SAFE_BUT_PASSIVE`.
- Accepter uniquement `blocked_capability=objective_conversion_after_safe_stop`.
- Exiger `ready_for_objective_conversion_review=true`.
- Exiger `ready_for_m2_or_m3=false`.
- Refuser `a33_ready=true`.
- Refuser `support>0`.
- Refuser `status=CONFIRMED` ou `status=REFUTED`.
- Refuser toute frontier sans `desired_hypothesis_families`, `requested_experiment_styles`, ou `scientific_questions`.
- Ne pas ecrire A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.objective_conversion_frontier_validator `
  --objective-conversion-frontier-records diagnostics\p2\objective_conversion_frontier_records.json `
  --out diagnostics\p2\objective_conversion_frontier_validation.json
```

Resume du run :

- objective_conversion_frontiers_seen = 1
- objective_conversion_reviews_accepted = 1
- objective_conversion_reviews_rejected = 0
- rejection_reasons = []
- ready_for_objective_conversion_review = true
- ready_for_m2_or_m3 = false
- objective_conversion_review_no_write = true
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Frontier acceptee :

```json
{
  "frontier_id": "p2g1::bp35::terminal_safe_but_passive::objective_conversion",
  "frontier_type": "OBJECTIVE_CONVERSION_FRONTIER",
  "frontier_reason": "TERMINAL_SAFE_BUT_PASSIVE",
  "blocked_capability": "objective_conversion_after_safe_stop",
  "objective_conversion_review_accepted": true,
  "objective_conversion_review_target": "objective_conversion_frontier_review",
  "ready_for_m2_or_m3": false,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY",
  "truth_status": "NOT_EVALUATED_BY_P2"
}
```

Lecture :

P2.G2 confirme que la frontier safe-but-passive est recevable comme objet de review objective-conversion. Le validateur ne transforme pas encore cette frontier en handoff M2/M3 : il joue uniquement la douane no-write, verifie que le dossier porte des questions et styles d'experience, et garde la separation entre utilite policy et verdict scientifique.

Suite naturelle :

- P2.G3 - Objective-conversion handoff schema vers `M2_OR_M3`, toujours stocke dans `diagnostics/p2/`.
- M2.G1/M3.G1 seulement apres ce schema, et toujours candidate-only.

## P2.G3 - Objective-conversion handoff schema

Objectif :

Transformer la review P2.G2 acceptee en requete objective-conversion candidate-only pour la branche M2/M3, sans ecrire dans M2/M3.

Entree :

- `diagnostics/p2/objective_conversion_frontier_validation.json`

Sortie :

- `diagnostics/p2/objective_conversion_handoff_requests.json`

Contrat :

- Produire `handoff_type=OBJECTIVE_CONVERSION_FRONTIER_REQUEST`.
- Produire `target=M2_OR_M3`.
- Produire `target_modules=["M2.G1","M3.G1"]`.
- Porter `source_frontier_id=p2g1::bp35::terminal_safe_but_passive::objective_conversion`.
- Porter `frontier_type=OBJECTIVE_CONVERSION_FRONTIER`.
- Porter `frontier_reason=TERMINAL_SAFE_BUT_PASSIVE`.
- Porter `blocked_capability=objective_conversion_after_safe_stop`.
- Conserver les familles d'hypotheses et styles d'experiences valides par P2.G2.
- Garder `ready_for_direct_downstream_write=false`.
- Ne pas ecrire A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.objective_conversion_handoff_schema `
  --objective-conversion-frontier-validation diagnostics\p2\objective_conversion_frontier_validation.json `
  --out diagnostics\p2\objective_conversion_handoff_requests.json
```

Resume du run :

- source_objective_conversion_reviews_accepted = 1
- objective_conversion_handoff_requests = 1
- handoff_type = `OBJECTIVE_CONVERSION_FRONTIER_REQUEST`
- target = `M2_OR_M3`
- target_modules = `M2.G1`, `M3.G1`
- ready_for_m2_or_m3_objective_conversion_branch = true
- ready_for_direct_downstream_write = false
- a33_ready = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`
- wrong_confirmations = 0

Requete produite :

```json
{
  "request_id": "p2g3::bp35-0a0ad940::objective_conversion::001",
  "handoff_type": "OBJECTIVE_CONVERSION_FRONTIER_REQUEST",
  "target": "M2_OR_M3",
  "target_modules": ["M2.G1", "M3.G1"],
  "source_frontier_id": "p2g1::bp35::terminal_safe_but_passive::objective_conversion",
  "frontier_type": "OBJECTIVE_CONVERSION_FRONTIER",
  "frontier_reason": "TERMINAL_SAFE_BUT_PASSIVE",
  "blocked_capability": "objective_conversion_after_safe_stop",
  "requested_hypothesis_families": [
    "post_safe_stop_objective_conversion",
    "subgoal_target_reselection",
    "objective_readiness_condition",
    "terminal_safe_sequence_search"
  ],
  "requested_experiment_styles": [
    "stop_state_action_matrix",
    "post_safe_stop_short_sequence_probe",
    "relation_target_ablation_after_safe_stop",
    "objective_completion_vs_relation_progress_discriminator"
  ],
  "ready_for_direct_downstream_write": false,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY",
  "truth_status": "NOT_EVALUATED_BY_P2"
}
```

Matrice initiale suggeree :

- base_state_family = `terminal_safe_stop_or_avoidance_state`
- single_step_actions = `ACTION3`, `ACTION4`, `ACTION6`
- short_sequences = `ACTION3,ACTION4`, `ACTION4,ACTION3`, `ACTION6,ACTION3`, `ACTION6,ACTION4`
- controls = `hold_or_stop_state`, `relation_progress_policy`
- success_metrics = `objective_completion_signal`, `levels_completed_after_rollout`, `terminal_adjusted_progress_after_stop`
- diagnostic_metrics = `relation_delta_after_stop`, `terminal_reentry_rate`, `changed_pixels`

Lecture :

P2.G3 ouvre le canal objectif-conversion vers M2/M3 sans toucher aux artefacts M2/M3. La requete dit : "voici la frontier safe-but-passive validee, voici les familles d'hypotheses et les styles d'experiences attendus". Elle ne dit pas : "execute maintenant" ni "confirme une mecanique". Le prochain module autorise a transformer cette requete en hypotheses falsifiables est `M2.G1`.

Suite naturelle :

- M2.G1 - Objective-conversion hypothesis generator.
- M3.G1 - Stop-state objective-conversion experiment planner/executor, apres hypotheses M2.G1.

## P2.G4 - Risk-aware post-stop policy frontier record

Objectif :

Transformer le resultat P3.G4 `RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY` en frontier P2 explicite : une policy post-stop est maintenant safe, utile OOS et risk-aware, mais elle ne convertit toujours pas ce progres en completion objectif.

Entree :

- `diagnostics/p3/risk_targeted_contextual_post_stop_policy_validation.json`

Sortie :

- `diagnostics/p2/risk_aware_post_stop_frontier_records.json`

Contrat :

- Declencher uniquement si P3.G4 a bien :
  - `policy_utility_status=RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY`
  - `execution_performed=true`
  - `source_cells_rerun=true`
  - `adapter_relearned=false`
  - `selection_uses_risk_targeted_candidate_outcomes=false`
  - `terminal_rate=0.0`
  - utilite positive vs hold et `ACTION6`
  - risque static-extension reproduit et evite par le selecteur
  - `objective_completion_signal=false`
- Produire `frontier_type=RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER`.
- Produire `frontier_reason=RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION`.
- Marquer `blocked_capability=objective_completion_after_risk_aware_safe_conversion`.
- Garder `ready_for_risk_aware_objective_frontier_review=true`.
- Garder `ready_for_m2_or_m3=false` et `ready_for_direct_downstream_write=false`.
- Ne pas ecrire A40/M2/M3/A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.risk_aware_post_stop_frontier_records `
  --p3-risk-targeted-validation diagnostics\p3\risk_targeted_contextual_post_stop_policy_validation.json `
  --out diagnostics\p2\risk_aware_post_stop_frontier_records.json
```

Resume du run :

- source_policy_utility_status = `RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY`
- frontier_records = 1
- frontier_record_status = `RISK_AWARE_POST_STOP_FRONTIER_RECORDED_CANDIDATE_ONLY`
- frontier_type = `RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER`
- frontier_reason = `RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION`
- blocked_capability = `objective_completion_after_risk_aware_safe_conversion`
- terminal_safety_observed = true
- oos_utility_observed = true
- risk_aware_selection_observed = true
- objective_completion_signal = false
- ready_for_risk_aware_objective_frontier_review = true
- ready_for_m2_or_m3 = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`

Evidence portee :

- P3.G4 contextual terminal rate = 0.0.
- P3.G4 contextual mean terminal-adjusted progress = 142.857143.
- Hold mean terminal-adjusted progress = 135.0.
- `ACTION6` mean terminal-adjusted progress = 140.0.
- Static `ACTION6,ACTION3` terminal rate = 0.285714.
- Static `ACTION6,ACTION4` terminal rate = 0.285714.
- Static extension terminal options = 4.
- Static extension terminal safe-stops = 2.
- Unsafe extension options avoided by selector = 4.
- Objective completion runs = 0.

Hypotheses bloquees transmises au dossier :

- `proxy_progress_not_completion_condition`
- `objective_readiness_detector_missing`
- `terminal_commit_or_submit_action_missing`
- `goal_representation_missing_beyond_safe_progress`
- `conversion_state_useful_but_not_completion_trigger`

Lecture :

P2.G4 ne demande pas une nouvelle optimisation de gate. Il formalise la frontier decouverte par P3.G4 : la policy post-stop sait choisir entre fallback safe et extension risquee, transporte une utilite OOS, et evite des reentrees terminales reelles. Le blocage restant n'est donc plus la safety terminale ni la selection risk-aware ; c'est la transformation de ce progres safe en completion objectif.

Suite naturelle :

- P2.G5 - Risk-aware frontier validator/schema no-write, si l'on veut ouvrir explicitement la prochaine branche M2/M3.
- Puis M2/M3 devront produire des hypotheses falsifiables sur objective-readiness, commit action, et representation du but, sans incrementer `support`.

## P2.G5 - Risk-aware objective frontier validator/schema

Objectif :

Valider la frontier P2.G4 comme recevable pour une branche M2/M3 centree sur la completion objectif apres conversion safe/risk-aware, puis produire une requete candidate-only stockee en P2. P2.G5 ne genere pas encore d'hypotheses M2 et n'ecrit jamais dans M2/M3/A32/A33.

Entree :

- `diagnostics/p2/risk_aware_post_stop_frontier_records.json`

Sortie :

- `diagnostics/p2/risk_aware_objective_frontier_handoff_requests.json`

Contrat :

- Accepter uniquement `frontier_type=RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER`.
- Accepter uniquement `frontier_reason=RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION`.
- Exiger `blocked_capability=objective_completion_after_risk_aware_safe_conversion`.
- Exiger :
  - `terminal_rate=0.0`
  - gain positif vs hold
  - gain positif vs `ACTION6`
  - static extension risk reproduit
  - unsafe extensions evitees par le selecteur
  - `objective_completion_signal=false`
  - `objective_completion_runs=0`
  - `adapter_relearned=false`
  - `source_cells_rerun=true`
  - `selection_uses_risk_targeted_candidate_outcomes=false`
- Exiger `ready_for_risk_aware_objective_frontier_review=true`.
- Exiger `ready_for_m2_or_m3=false` en entree.
- Refuser `support>0`, `a33_ready=true`, `CONFIRMED`, `REFUTED`, ou tout write A40/M2/M3/A32/A33.
- Produire `handoff_type=RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST`.
- Produire `target=M2_OR_M3`, `target_modules=["M2.G2"]`.
- Garder `ready_for_direct_downstream_write=false`.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P2`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p2.risk_aware_objective_frontier_handoff_schema `
  --risk-aware-frontier-records diagnostics\p2\risk_aware_post_stop_frontier_records.json `
  --out diagnostics\p2\risk_aware_objective_frontier_handoff_requests.json
```

Resume du run :

- risk_aware_frontiers_seen = 1
- risk_aware_objective_reviews_accepted = 1
- risk_aware_objective_reviews_rejected = 0
- risk_aware_objective_handoff_requests = 1
- rejection_reasons = []
- handoff_type = `RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST`
- target = `M2_OR_M3`
- target_modules = `M2.G2`
- suggested_followup_modules = `M3.G5`
- ready_for_m2_or_m3_risk_aware_objective_branch = true
- ready_for_direct_downstream_write = false
- a33_ready = false
- a40_write_performed = false
- m2_write_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P2`

Requete produite :

```json
{
  "request_id": "p2g5::bp35-0a0ad940::risk_aware_objective_completion::001",
  "handoff_type": "RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST",
  "target": "M2_OR_M3",
  "target_modules": ["M2.G2"],
  "source_frontier_id": "p2g4::bp35::risk_aware_post_stop_no_objective_completion",
  "frontier_type": "RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER",
  "frontier_reason": "RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION",
  "blocked_capability": "objective_completion_after_risk_aware_safe_conversion",
  "ready_for_risk_aware_objective_hypothesis_generation": true,
  "ready_for_direct_downstream_write": false,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY",
  "truth_status": "NOT_EVALUATED_BY_P2"
}
```

Matrice initiale suggeree :

- base_state_family = `risk_aware_terminal_safe_post_stop_conversion_state`
- source_policy_options = `hold_or_stop_state`, `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4`, `contextual_post_stop_conversion_policy`
- readiness_feature_candidates = `sampling_family`, `terminal_horizon_remaining`, `terminal_horizon_band`, `hold_baseline_terminal_adjusted_progress`, `hold_baseline_band`, `relation_delta_after_stop`, `new_relation_states`, `changed_pixels`, `global_configuration_signature`
- post_conversion_commit_action_candidates = `ACTION1`, `ACTION2`, `ACTION3`, `ACTION4`, `ACTION5`, `ACTION6`
- controls = `hold_or_stop_state`, `ACTION6_only`, `frozen_contextual_selector`, `always_extension_static_policy`
- success_metrics = `objective_completion_signal`, `levels_completed_after_rollout`
- safety_metrics = `terminal_reentry_rate`, `terminal_adjusted_progress_after_stop`
- discriminator_metrics = `proxy_progress_without_completion`, `objective_readiness_precision`, `commit_action_delta_vs_selector`

Lecture :

P2.G5 ouvre le canal M2.G2 sans consommer la frontier. La requete dit : "la policy sait etre safe, utile OOS et risk-aware, mais ne complete pas ; produire maintenant des hypotheses falsifiables sur readiness, commit action, representation du but, et discriminateur proxy-vs-completion". Elle ne dit pas que le selecteur est une solution, et elle ne cree aucun write aval.

Suite naturelle :

- M2.G2 - Objective-readiness / commit-action hypothesis generator.
- Puis M3.G5 seulement apres hypotheses M2.G2, pour tester completion-readiness et commit actions sur les etats post-stop safe/risk-aware.
