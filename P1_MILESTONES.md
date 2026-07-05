# P1 Milestones - Agent Policy Probes

Derniere mise a jour : 2026-06-20

P1 ne revise pas la science. P1 teste uniquement si une regle candidate-only issue de M3/A32 ameliore le comportement agentique en boucle fermee, sans LLM, sans world model, sans A33.

## Principes fixes

- P1 peut lire un dossier candidate-only explicitement marque pret pour probe agentique.
- P1 ne lit pas A33 et n'ecrit jamais A33.
- P1 ne confirme pas, ne refute pas, et ne modifie aucun support scientifique.
- Toute policy candidate reste experimentale : `EXPERIMENTAL_POLICY_CANDIDATE_ONLY`.
- Tous les artefacts P1 gardent `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P1_AGENT_PROBE`, `wrong_confirmations=0`.
- `candidate_policy_counted_as_confirmation=false` doit rester vrai meme si la policy candidate bat le baseline.

## Etat

| Milestone | Statut | Artefacts | Resultat |
| --- | --- | --- | --- |
| P1.1 - bp35 closed-loop SAGE candidate-policy probe | Fait | `theory/p1/bp35_sage_candidate_policy_probe.py`, `tests/test_p1_bp35_sage_candidate_policy_probe.py`, `diagnostics/p1/bp35_sage_candidate_policy_probe.json` | Candidate policy > baseline sur progress proxy, ACTION6 utiles, et evitement des args failure-like. Aucun verdict. |
| P1.2 - Multi-budget deterministic seeds | Fait | `diagnostics/p1/bp35_sage_candidate_policy_probe_matrix.json` | Candidate policy > baseline sur 18/18 runs, budgets 4/8/12/16/24/32 et seeds 0/1/2. Aucun verdict. |
| P1.3 - ACTION4-only ablation | Fait | `diagnostics/p1/bp35_sage_candidate_policy_probe_matrix.json` | Candidate policy > ACTION4-only sur 18/18 runs. Le gain ne vient pas seulement de repositionner avec ACTION4. |
| P1.4 - Candidate policy utility handoff | Fait | `theory/p1/bp35_candidate_policy_utility_handoff.py`, `tests/test_p1_candidate_policy_utility_handoff.py`, `diagnostics/p1/bp35_candidate_policy_utility_handoff.json` | Dossier d'utilite agentique candidate-only produit pour revue ulterieure, sans confirmation mecanistique. |
| P1.5 - Patch-similarity-only ablation | Fait | `diagnostics/p1/bp35_sage_candidate_policy_ablation_matrix.json` | Patch-similarity-only bat la candidate ACTION4+patch sur 18/18 runs. La dependance stricte a ACTION4 n'est pas soutenue par ce proxy. |
| P1.6 - Hard stale/repetition guard | Fait | `diagnostics/p1/bp35_sage_candidate_policy_stale_guard_matrix.json` | Le guard dur reduit les repetitions mais degrade le progress_proxy. Stale doit dependre de l'absence d'effet utile, pas seulement de "deja joue". |
| P1.7 - Effect-aware soft stale guard | Fait | `diagnostics/p1/bp35_sage_candidate_policy_soft_stale_guard_matrix.json` | Le soft guard preserve le gain patch-only et bat le hard guard sur 15/18 runs. Les repetitions utiles restent autorisees. |
| P1.8 - Conditional ACTION4 refresh | Fait | `diagnostics/p1/bp35_sage_candidate_policy_conditional_refresh_matrix.json` | Le refresh conditionnel preserve le signal soft-stale/patch-only, mais ne se declenche pas sur les budgets testes. ACTION4 reste un refresh sous saturation, pas une action obligatoire. |
| P1.9 - Frontier trigger after exhausted refresh | Fait | `theory/p1/bp35_policy_frontier_trigger.py`, `tests/test_p1_bp35_policy_frontier_trigger.py`, `diagnostics/p1/bp35_policy_frontier_trigger.json` | Aucun trigger frontier reel sur P1.8, car pas de saturation observee. Fixture synthetique de saturation valide le trigger sans produire de handoff scientifique reel. |
| P1.10 - Conditional movement refresh selector | Fait | `diagnostics/p1/bp35_sage_candidate_policy_movement_refresh_matrix.json` | Le privilege ACTION4 est retire : ACTION3/ACTION4/ACTION1/ACTION2 sont candidats refresh sous saturation. Aucun mouvement n'est declenche sur les budgets testes, car ACTION6 reste productif. |
| P1.11 - Generalized movement-refresh frontier trigger | Fait | `theory/p1/bp35_policy_frontier_trigger.py`, `tests/test_p1_bp35_policy_frontier_trigger.py`, `diagnostics/p1/bp35_movement_policy_frontier_trigger.json` | Le trigger frontier consomme maintenant P1.10 et vise `conditional_movement_refresh`. Aucun trigger reel, fixture synthetique valide le cas epuise. |

## P1.1 - bp35 closed-loop SAGE candidate-policy probe

Objectif :

Tester si la regle candidate-only condensee par M3.24 ameliore le comportement reel de l'agent sur bp35.

Entree :

- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_probe.json`

Contrat :

- Deux conditions executees : baseline sans regle M3.24, candidate_policy avec regle M3.24.
- Pas de LLM.
- Pas de world model.
- Pas de lecture/ecriture A33.
- Replay et execution dans l'environnement bp35 offline.
- La candidate policy utilise ACTION4 comme repositionnement candidat, puis selectionne ACTION6 par similarite de patch avec les succes connus.
- Les args failure-like ou boundary-like sont penalises, notamment `{x:18,y:0}` et `{x:30,y:0}`.
- Les args success-like connus incluent `{x:12,y:0}`, `{x:24,y:0}`, `{x:30,y:12}`, `{x:36,y:12}`, `{x:42,y:12}`, `{x:48,y:12}`.
- La comparaison agentique n'est jamais interpretee comme confirmation scientifique.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_sage_candidate_policy_probe `
  --scope-consolidation diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --budget 8
```

Resume du run :

- conditions_run = 2
- budget_per_condition = 8
- candidate_policy_probe_ready = true
- candidate_policy_status = `EXPERIMENTAL_POLICY_CANDIDATE_ONLY`
- a33_ready = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P1_AGENT_PROBE`
- wrong_confirmations = 0

Baseline :

- policy_steps = 8
- progress_proxy = 33.0
- action6_steps = 2
- useful_action6_steps = 2
- success_like_action6_args_selected = 0
- failure_like_action6_args_selected = 2
- useful_new_states = 5
- useful_repositioning_steps = 3
- cycles_detected = 0

Candidate policy :

- policy_steps = 8
- progress_proxy = 60.0
- action6_steps = 4
- useful_action6_steps = 4
- success_like_action6_args_selected = 4
- failure_like_action6_args_selected = 0
- useful_new_states = 8
- useful_repositioning_steps = 4
- cycles_detected = 0

Comparaison :

- candidate_policy_better_than_baseline_on_any_axis = true
- improved_axes = `progress_proxy`, `useful_action6_steps`, `failure_like_action6_args_selected`
- candidate_progress_proxy_delta = 27.0
- candidate_useful_action6_delta = 2
- candidate_failure_like_selection_delta = -2
- candidate_policy_counted_as_confirmation = false

Lecture :

La regle candidate-only issue de M3.24 ameliore le comportement agentique dans ce probe bp35 court : elle force une alternance repositionnement ACTION4 puis ACTION6 success-like, evite les args boundary/failure-like choisis par le baseline, et produit plus de signaux utiles. Ce resultat reste une performance de policy experimentale, pas une confirmation scientifique.

## P1.2 - Multi-budget deterministic seeds

Objectif :

Verifier que le gain P1.1 ne depend pas uniquement du budget 8 ou d'un tie-break chanceux.

Entree :

- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_probe_matrix.json`

Contrat :

- Budgets testes : 4, 8, 12, 16, 24, 32.
- Tie-break seeds deterministes : 0, 1, 2.
- Conditions executees a chaque run : baseline, ACTION4-only ablation, candidate_policy M3.24.
- Pas de LLM.
- Pas de world model.
- Pas de lecture/ecriture A33.
- Resultat de policy non interprete comme verdict scientifique.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_sage_candidate_policy_probe `
  --matrix `
  --scope-consolidation diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p1\bp35_sage_candidate_policy_probe_matrix.json `
  --budgets 4 8 12 16 24 32 `
  --tie-break-seeds 0 1 2
```

Resume du run :

- budget_runs = 18
- candidate_beats_baseline_runs = 18
- candidate_beats_baseline_ratio = 1.0
- candidate_mean_progress_delta_vs_baseline = 57.2222
- candidate_mean_failure_like_selection_delta_vs_baseline = -4.4444
- robust_candidate_policy_utility_signal = true
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P1_AGENT_PROBE`
- wrong_confirmations = 0

Lecture :

Le signal P1.1 se generalise sur les budgets courts et longs testes. La candidate policy reste meilleure que la baseline sur les 18 variantes deterministes, avec un gain moyen positif de progress_proxy et une forte reduction des args failure-like/boundary-like. Ce resultat reste une utilite agentique candidate-only.

## P1.3 - ACTION4-only ablation

Objectif :

Verifier que le gain ne vient pas seulement du fait d'utiliser ACTION4 comme repositionnement, mais aussi de la selection ACTION6 par similarite de patch.

Conditions :

- baseline : exploration deterministe sans regle M3.24.
- ACTION4-only : utilise ACTION4 comme repositionnement, puis choisit ACTION6 sans scoring patch-similar.
- candidate_policy : utilise ACTION4 puis choisit ACTION6 par similarite avec les succes M3/A32.

Resume du run :

- candidate_beats_action4_only_runs = 18
- candidate_beats_action4_only_ratio = 1.0
- action4_only_mean_progress_delta_vs_baseline = 52.2222
- candidate_mean_progress_delta_vs_baseline = 57.2222
- patch_similarity_attribution_signal_candidate_only = true
- candidate_policy_counted_as_confirmation = false
- ablation_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

ACTION4-only est deja utile : le repositionnement explique une grande partie du gain. Mais la candidate policy bat ACTION4-only sur 18/18 runs, ce qui isole un signal additionnel attribuable a la selection patch-similar ACTION6. Ce signal est une attribution experimentale de policy, pas une confirmation mecanistique.

Suite naturelle :

- P1.9 - Frontier trigger quand soft-stale/refresh n'a plus d'ACTION6 utile.
- P2.1 - Extraction de frontier depuis un rollout policy sature.

## P1.4 - Candidate policy utility handoff

Objectif :

Transmettre a une revue ulterieure un dossier d'utilite agentique, sans demander de confirmation mecanistique.

Entrees :

- `diagnostics/p1/bp35_sage_candidate_policy_probe.json`
- `diagnostics/p1/bp35_sage_candidate_policy_probe_matrix.json`
- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Sortie :

- `diagnostics/p1/bp35_candidate_policy_utility_handoff.json`

Contrat :

- Handoff type = `AGENTIC_UTILITY_CANDIDATE_ONLY`.
- Agentic utility status = `SUPPORTED_CANDIDATE_ONLY`.
- La performance policy n'est pas une confirmation mecanistique.
- La performance policy n'est pas une readiness A33.
- P1.4 n'ecrit pas A32/A33.
- `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P1_AGENT_PROBE`.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_candidate_policy_utility_handoff `
  --probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --matrix diagnostics\p1\bp35_sage_candidate_policy_probe_matrix.json `
  --scope diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p1\bp35_candidate_policy_utility_handoff.json
```

Resume du run :

- handoff_type = `AGENTIC_UTILITY_CANDIDATE_ONLY`
- agentic_utility_status = `SUPPORTED_CANDIDATE_ONLY`
- handoffs_produced = 1
- handoffs_rejected = 0
- candidate_beats_baseline_runs = 18
- candidate_beats_action4_only_runs = 18
- patch_similarity_attribution_signal_candidate_only = true
- policy_result_counted_as_mechanistic_confirmation = false
- a33_ready = false
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

P1.4 separe explicitement deux choses : la regle n'est pas confirmee comme mecanisme, mais elle a une utilite agentique robuste dans le probe closed-loop. C'est un dossier d'usage, pas une inscription scientifique.

## P1.5 - Patch-similarity-only ablation

Objectif :

Tester si la selection ACTION6 patch-similar depend vraiment d'un repositionnement ACTION4 obligatoire.

Conditions :

- baseline : exploration deterministe sans regle M3.24.
- ACTION4-only : utilise ACTION4 comme repositionnement, puis choisit ACTION6 sans scoring patch-similar.
- patch-similarity-only : choisit ACTION6 par scoring patch-similar, sans utiliser ACTION4.
- candidate_policy : utilise ACTION4 puis choisit ACTION6 par similarite avec les succes M3/A32.

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_ablation_matrix.json`

Resume du run :

- budget_runs = 18
- conditions_per_run = 4
- candidate_beats_baseline_runs = 18
- candidate_beats_action4_only_runs = 18
- patch_similarity_only_runs = 18
- candidate_beats_patch_similarity_only_runs = 0
- candidate_beats_patch_similarity_only_ratio = 0.0
- patch_similarity_only_mean_progress_delta_vs_baseline = 65.2222
- candidate_mean_progress_delta_vs_baseline = 57.2222
- repositioning_context_dependency_signal_candidate_only = false
- patch_similarity_only_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

Le resultat attendu "candidate_policy > patch-similarity-only" n'est pas observe. Au contraire, patch-similarity-only bat la candidate ACTION4+patch sur 18/18 runs selon le progress_proxy actuel. La bonne lecture n'est donc pas "patch similarity apres repositionnement seulement", mais plutot : la selection patch-similar ACTION6 est deja tres utile en policy, et l'obligation rigide d'intercaler ACTION4 peut couter du budget.

Point de vigilance :

Patch-similarity-only repete parfois des ACTION6 success-like deja utiles. Avant de le transformer en runtime policy, il faut ajouter une garde stale/repetition et une extraction de frontier quand la liste d'affordances patch-similar cesse de produire de nouveaux etats utiles.

## P1.6 - Hard stale/repetition guard

Objectif :

Tester une premiere garde contre les repetitions ACTION6 patch-similar.

Conditions :

- baseline
- ACTION4-only
- patch-similarity-only
- patch-similarity-stale-guard
- candidate_policy ACTION4+patch

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_stale_guard_matrix.json`

Regle testee :

- Choisir ACTION6 patch-similar.
- Exclure failure-like et boundary-like.
- Exclure tout ACTION6 arg deja joue.
- Si aucun ACTION6 frais et safe n'existe, fallback exploration sans ACTION4 ni ACTION6.

Resume du run :

- budget_runs = 18
- conditions_per_run = 5
- stale_guard_runs = 18
- stale_guard_beats_patch_similarity_only_runs = 0
- stale_guard_beats_patch_similarity_only_ratio = 0.0
- stale_guard_mean_progress_delta_vs_baseline = -17.4444
- stale_guard_mean_repeated_action6_delta_vs_patch_similarity_only = -10.3333
- stale_guard_repetition_nonincrease_signal_candidate_only = true
- stale_guard_repetition_reduction_signal_candidate_only = false
- stale_guard_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

P1.6 est un bon resultat negatif. Le guard dur reduit fortement les repetitions, mais il degrade le progress_proxy parce qu'il interdit aussi des repetitions ACTION6 qui restent utiles selon les metriques actuelles. A budget 8, patch-similarity-only atteint progress_proxy=64 avec 8 ACTION6 utiles et 2 repetitions ; stale-guard tombe a 48 avec 6 ACTION6 utiles et 0 repetition.

Conclusion :

Il ne faut pas definir stale comme "arg deja joue". Il faut definir stale comme "arg repete qui ne produit plus de nouvel etat utile", ou "arg repete dont le signal local/objet est sature". La prochaine policy doit donc etre un soft stale guard, pas un hard no-repeat guard.

## P1.7 - Effect-aware soft stale guard

Objectif :

Tester une garde stale basee sur l'effet observe, pas seulement sur l'identite de l'arg.

Conditions :

- baseline
- ACTION4-only
- patch-similarity-only
- hard stale guard
- soft stale guard
- candidate_policy ACTION4+patch

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_soft_stale_guard_matrix.json`

Regle testee :

- Choisir ACTION6 patch-similar.
- Exclure failure-like et boundary-like.
- Autoriser un ACTION6 arg repete si sa derniere occurrence a encore produit un effet utile.
- Bloquer un ACTION6 arg repete seulement si `consecutive_no_new_effects >= 1`.
- Si aucun ACTION6 effectif n'existe, fallback exploration sans ACTION4 ni ACTION6.

Memoire d'effet par arg :

```json
{
  "times_selected": 3,
  "new_state_count": 2,
  "useful_effect_count": 3,
  "consecutive_no_new_effects": 0,
  "last_local_patch_signal": 2.0,
  "last_object_positions_signal": 4.0,
  "last_effect_useful": true
}
```

Resume du run :

- budget_runs = 18
- conditions_per_run = 6
- soft_stale_guard_runs = 18
- soft_stale_guard_mean_progress_delta_vs_baseline = 65.2222
- patch_similarity_only_mean_progress_delta_vs_baseline = 65.2222
- soft_stale_guard_beats_hard_stale_guard_runs = 15
- soft_stale_guard_beats_hard_stale_guard_ratio = 0.8333
- soft_stale_guard_beats_patch_similarity_only_runs = 0
- soft_stale_guard_mean_repeated_action6_delta_vs_patch_similarity_only = 0.0
- soft_stale_guard_mean_sterile_repetition_delta_vs_patch_similarity_only = 0.0
- soft_stale_guard_preserves_patch_only_progress_signal_candidate_only = true
- soft_stale_guard_sterile_repetition_nonincrease_signal_candidate_only = true
- soft_stale_guard_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

P1.7 corrige l'erreur de P1.6. Le soft guard ne bloque pas les repetitions encore utiles ; il preserve donc la performance de patch-similarity-only. A budget 8, patch-only et soft-stale atteignent tous les deux progress_proxy=64 avec 8 ACTION6 utiles, 6 args uniques, 2 repetitions, et 0 repetition sterile.

Interpretation :

Il n'y avait pas encore de repetition sterile a corriger dans ces rollouts, donc soft-stale ne peut pas battre patch-only sur le proxy actuel. Mais il formalise le bon comportement runtime : autoriser les repetitions productives, bloquer uniquement les repetitions sans nouvel effet, puis preparer un refresh ACTION4 ou une frontier si l'espace ACTION6 effectif est epuise.

## P1.8 - Conditional ACTION4 refresh

Objectif :

Tester ACTION4 comme refresh/repositionnement uniquement quand la selection ACTION6 patch-similar effect-aware n'a plus d'action effective disponible.

Conditions :

- baseline
- ACTION4-only
- patch-similarity-only
- hard stale guard
- soft stale guard
- conditional ACTION4 refresh
- candidate_policy ACTION4+patch

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_conditional_refresh_matrix.json`

Regle testee :

- Choisir ACTION6 patch-similar en priorite.
- Autoriser les repetitions ACTION6 tant qu'elles produisent encore un effet utile.
- Exclure les args failure-like et boundary-like.
- Si aucun ACTION6 effectif n'est disponible, executer ACTION4 comme refresh.
- Re-enumerer ensuite les ACTION6 live.
- Si ACTION4 n'est pas disponible ou vient deja d'etre joue, fallback exploration sans ACTION4/ACTION6.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_sage_candidate_policy_probe `
  --matrix `
  --include-conditional-refresh `
  --scope-consolidation diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p1\bp35_sage_candidate_policy_conditional_refresh_matrix.json `
  --budgets 4 8 12 16 24 32 `
  --tie-break-seeds 0 1 2
```

Resume du run :

- budget_runs = 18
- conditions_per_run = 7
- conditional_refresh_runs = 18
- conditional_refresh_mean_progress_delta_vs_baseline = 65.2222
- soft_stale_guard_mean_progress_delta_vs_baseline = 65.2222
- patch_similarity_only_mean_progress_delta_vs_baseline = 65.2222
- conditional_refresh_beats_patch_similarity_only_runs = 0
- conditional_refresh_beats_soft_stale_guard_runs = 0
- conditional_refresh_total_triggers = 0
- conditional_refresh_runs_with_triggers = 0
- conditional_refresh_mean_new_action6_affordances_after_refresh = 0.0
- conditional_refresh_preserves_soft_stale_progress_signal_candidate_only = true
- conditional_refresh_triggered_in_any_run_candidate_only = false
- conditional_refresh_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P1_AGENT_PROBE`

Lecture :

P1.8 est un resultat neutre mais utile. La policy conditionnelle n'impose plus ACTION4, donc elle preserve exactement le signal patch-only/soft-stale. Sur les budgets 4/8/12/16/24/32 et seeds 0/1/2, elle ne declenche jamais ACTION4 : cela signifie que l'espace ACTION6 patch-similar effectif n'est pas encore epuise dans ces rollouts. ACTION4 reste donc correctement positionnee comme refresh sous saturation, pas comme alternance obligatoire.

Conclusion :

Le runtime policy actuel peut etre formule ainsi : exploiter ACTION6 patch-similar, autoriser les repetitions productives, declencher ACTION4 seulement si aucun ACTION6 effectif n'est disponible, puis produire une frontier si le refresh ne debloque rien. La prochaine etape propre est P1.9, avec un trigger de frontier quand soft-stale plus refresh arrive vraiment a saturation.

## P1.9 - Frontier trigger after exhausted refresh

Objectif :

Ajouter le mecanisme qui transforme une saturation runtime en frontier scientifique candidate-only : si la policy n'a plus d'ACTION6 effectif, essaie ACTION4 comme refresh, puis ne force pas si le refresh ne debloque rien.

Entree :

- `diagnostics/p1/bp35_sage_candidate_policy_conditional_refresh_matrix.json`

Sortie :

- `diagnostics/p1/bp35_policy_frontier_trigger.json`

Contrat :

- P1.9 lit uniquement P1.8.
- P1.9 ne lit pas A33.
- P1.9 n'ecrit pas A32/A33/M3.
- P1.9 ne confirme pas, ne refute pas, et ne modifie aucun support scientifique.
- Une frontier reelle n'est prete pour P2 que si la saturation est observee dans un rollout reel.
- La fixture synthetique valide seulement le contrat du trigger et n'est pas un handoff scientifique.

Regle de trigger :

```json
{
  "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_REFRESH",
  "exhausted_policy": "conditional_action4_refresh",
  "policy_result_counted_as_scientific_verdict": false,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY",
  "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE"
}
```

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_policy_frontier_trigger `
  --conditional-refresh-matrix diagnostics\p1\bp35_sage_candidate_policy_conditional_refresh_matrix.json `
  --out diagnostics\p1\bp35_policy_frontier_trigger.json
```

Resume P1.9a - matrice reelle P1.8 :

- conditional_refresh_summaries_seen = 18
- conditional_refresh_total_triggers = 0
- conditional_refresh_runs_with_triggers = 0
- exhausted_refresh_runs = 0
- real_frontier_triggered = false
- real_ready_for_p2_frontier_extraction = false
- frontier_reason = `NO_REFRESH_SATURATION_OBSERVED`
- support = 0
- revision_status = `CANDIDATE_ONLY`

Resume P1.9b - fixture synthetique de saturation :

- conditional_refresh_total_triggers = 1
- exhausted_refresh_runs = 1
- synthetic_fixture_frontier_triggered = true
- synthetic_fixture_validates_trigger_logic = true
- frontier_reason = `NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_REFRESH`
- ready_for_p2_frontier_extraction = true dans la fixture
- synthetic_fixture_not_for_scientific_handoff = true
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

P1.9 est un bon resultat de controle. Sur les rollouts reels P1.8, aucune frontier n'est declenchee, parce que la policy n'a jamais epuise l'espace ACTION6 effectif et n'a jamais eu besoin d'ACTION4 comme refresh. La fixture synthetique montre en revanche que le trigger s'allume correctement dans le cas attendu : soft-stale epuise, ACTION4 essaye, aucune nouvelle affordance ACTION6 utile apres refresh.

Conclusion :

Le runtime sait maintenant distinguer deux situations : continuer a exploiter quand ACTION6 reste productif, ou produire une frontier si ACTION6 plus refresh n'ouvre plus rien. La prochaine etape propre est P2.1 : convertir une frontier P1 reelle, lorsqu'elle existe, en FrontierRecord exploitable par A40/M2/M3. Pour l'instant, il n'y a pas encore de frontier reelle dans bp35 P1.8.

## P1.10 - Conditional movement refresh selector

Objectif :

Retirer le privilege arbitraire donne a ACTION4 dans P1.8. ACTION4 reste un candidat refresh, mais ACTION3 et d'autres mouvements doivent pouvoir etre testes sous saturation.

Question :

Pourquoi ACTION4 seulement ? Reponse P1.10 : il ne faut pas figer ACTION4. La bonne abstraction est `refresh_action in {ACTION3, ACTION4, ACTION1, ACTION2}` sous saturation, avec ACTION6 patch-similar toujours prioritaire tant qu'il reste productif.

Conditions :

- baseline
- ACTION4-only
- patch-similarity-only
- hard stale guard
- soft stale guard
- conditional ACTION4 refresh
- conditional movement refresh
- candidate_policy ACTION4+patch

Sortie :

- `diagnostics/p1/bp35_sage_candidate_policy_movement_refresh_matrix.json`

Regle testee :

- Choisir ACTION6 patch-similar en priorite.
- Autoriser les repetitions ACTION6 tant qu'elles produisent encore un effet utile.
- Exclure les args failure-like et boundary-like.
- Si aucun ACTION6 effectif n'est disponible, choisir un refresh parmi `ACTION3`, `ACTION4`, `ACTION1`, `ACTION2`.
- Penaliser seulement l'action immediatement repetee et preferer le refresh le moins utilise.
- Re-enumerer ensuite les ACTION6 live.
- Ne jamais compter le choix d'un mouvement comme confirmation scientifique.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_sage_candidate_policy_probe `
  --matrix `
  --include-movement-refresh `
  --scope-consolidation diagnostics\m3\a32_requested_patch_similarity_scope_consolidation.json `
  --out diagnostics\p1\bp35_sage_candidate_policy_movement_refresh_matrix.json `
  --budgets 4 8 12 16 24 32 `
  --tie-break-seeds 0 1 2
```

Resume du run :

- budget_runs = 18
- conditions_per_run = 8
- movement_refresh_runs = 18
- movement_refresh_mean_progress_delta_vs_baseline = 65.2222
- soft_stale_guard_mean_progress_delta_vs_baseline = 65.2222
- patch_similarity_only_mean_progress_delta_vs_baseline = 65.2222
- movement_refresh_beats_patch_similarity_only_runs = 0
- movement_refresh_beats_soft_stale_guard_runs = 0
- movement_refresh_beats_conditional_action4_refresh_runs = 0
- movement_refresh_total_triggers = 0
- movement_refresh_runs_with_triggers = 0
- movement_refresh_actions_selected_counts = `{}`
- movement_refresh_preserves_soft_stale_progress_signal_candidate_only = true
- movement_refresh_triggered_in_any_run_candidate_only = false
- movement_refresh_counted_as_confirmation = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_P1_AGENT_PROBE`

Lecture :

P1.10 corrige l'abstraction sans changer le comportement observe sur ces rollouts. ACTION3 est maintenant un refresh candidat, mais aucun refresh n'est declenche parce que l'espace ACTION6 patch-similar reste productif sur tous les budgets/seeds testes. Le resultat important n'est donc pas "ACTION3 marche" ni "ACTION4 marche", mais : le runtime ne privilegie plus ACTION4 et ne paie aucun mouvement tant qu'ACTION6 continue a produire un effet.

Conclusion :

La hierarchie runtime devient plus propre : exploiter ACTION6 effectif, puis seulement sous saturation tester un mouvement refresh parmi plusieurs candidats, puis produire une frontier si aucun mouvement ne reouvre d'affordance. P1.9 devra ensuite consommer la version generalisee `conditional_movement_refresh` pour P2 quand une saturation reelle apparait.

## P1.11 - Generalized movement-refresh frontier trigger

Objectif :

Generaliser P1.9 pour consommer P1.10 plutot que l'ancien refresh ACTION4-only.

Entree :

- `diagnostics/p1/bp35_sage_candidate_policy_movement_refresh_matrix.json`

Sortie :

- `diagnostics/p1/bp35_movement_policy_frontier_trigger.json`

Contrat :

- P1.11 lit P1.10.
- P1.11 ne lit pas A33.
- P1.11 n'ecrit pas A32/A33/M3.
- P1.11 ne confirme pas, ne refute pas, et ne modifie aucun support scientifique.
- Une frontier reelle n'est prete pour P2 que si la saturation est observee dans un rollout reel.
- La fixture synthetique valide seulement le contrat du trigger et n'est pas un handoff scientifique.

Regle de trigger generalisee :

```json
{
  "frontier_reason": "NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH",
  "exhausted_policy": "conditional_movement_refresh",
  "refresh_candidates": ["ACTION3", "ACTION4", "ACTION1", "ACTION2"],
  "policy_result_counted_as_scientific_verdict": false,
  "support": 0,
  "revision_status": "CANDIDATE_ONLY",
  "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE"
}
```

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p1.bp35_policy_frontier_trigger `
  --refresh-mode movement `
  --movement-refresh-matrix diagnostics\p1\bp35_sage_candidate_policy_movement_refresh_matrix.json `
  --out diagnostics\p1\bp35_movement_policy_frontier_trigger.json
```

Resume P1.11a - matrice reelle P1.10 :

- refresh_mode = `movement`
- exhausted_policy = `conditional_movement_refresh`
- refresh_candidates = `ACTION3`, `ACTION4`, `ACTION1`, `ACTION2`
- refresh_summaries_seen = 18
- refresh_total_triggers = 0
- refresh_runs_with_triggers = 0
- exhausted_refresh_runs = 0
- real_frontier_triggered = false
- real_ready_for_p2_frontier_extraction = false
- frontier_reason = `NO_MOVEMENT_REFRESH_SATURATION_OBSERVED`
- support = 0
- revision_status = `CANDIDATE_ONLY`

Resume P1.11b - fixture synthetique de saturation movement-refresh :

- refresh_total_triggers = 1
- exhausted_refresh_runs = 1
- synthetic_fixture_frontier_triggered = true
- synthetic_fixture_validates_trigger_logic = true
- frontier_reason = `NO_EFFECTIVE_ACTION6_AFTER_SOFT_STALE_AND_CONDITIONAL_MOVEMENT_REFRESH`
- ready_for_p2_frontier_extraction = true dans la fixture
- synthetic_fixture_not_for_scientific_handoff = true
- support = 0
- revision_status = `CANDIDATE_ONLY`

Lecture :

P1.11 corrige la derniere trace du biais ACTION4 dans le trigger frontier. Le trigger reel reste eteint, parce que P1.10 n'observe toujours aucune saturation : ACTION6 reste productif et aucun mouvement refresh n'est selectionne. Mais la fixture prouve que si `conditional_movement_refresh` s'epuise sans rouvrir d'ACTION6 utile, P1 produira une frontier prete pour P2.

Conclusion :

P1 est maintenant coherent de bout en bout : la policy n'impose aucun mouvement, teste un refresh generique seulement sous saturation, et ne produit une frontier que si ACTION6 plus mouvement-refresh ne debloque rien. La prochaine etape propre est P2.1, avec un schema FrontierRecord compatible A40/M2/M3.
