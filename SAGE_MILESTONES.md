# SAGE milestones - closed-loop integration

Derniere mise a jour : 2026-07-19

SAGE orchestre les briques M1/M2/M3/P1 dans une boucle agentique. SAGE ne
confirme pas une mecanique, ne refute rien, et ne transforme jamais un resultat
de policy en support scientifique.

## Principes fixes

- `offline_trace_context` sert a l'observation, au diagnostic et au grounding.
- Les actions alternatives contrefactuelles ne sont autorisees que depuis un
  `live_env_context`.
- Une grille offline `grid_t` n'est jamais traitee comme un etat env complet.
- Les resultats LLM, WM, M3 et P1 restent candidate-only.
- A32/A33 restent les seuls lieux de revision/verdict scientifique.
- Tous les artefacts SAGE gardent `support=0`, `revision_status=CANDIDATE_ONLY`
  et `wrong_confirmations=0`.

## Etat

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| SAGE.0 - Known-game closed-loop scaffold | Fait | `theory/sage/known_game_closed_loop_scaffold.py`, `tests/test_sage_known_game_closed_loop_scaffold.py`, `diagnostics/sage/sage0_known_game_closed_loop_scaffold.json` | Boucle observation live -> contexte M2/M3/P1 -> action candidate -> env step -> log scientifique; offline counterfactual interdit; active counterfactual live autorise; support=0 |
| SAGE.1 - Known-game technical live run | Fait | `theory/sage/known_game_closed_loop_runner.py`, `theory/sage/live_prefix_counterfactual_collector.py`, `tests/test_sage_known_game_closed_loop_runner.py`, `tests/test_sage_live_prefix_counterfactual_collector.py`, `diagnostics/sage/sage1_known_game_closed_loop_results.json` | Runner bp35 reel budget 20; actions disponibles depuis `real_env_live_api`; aucune action illegale; replay prefix live exact pour ACTION3 alternatif; support=0 |
| SAGE.1b - Known-game policy loop guard | Fait | `theory/sage/policy_loop_guard.py`, `tests/test_sage_policy_loop_guard.py`, `diagnostics/sage/sage1b_policy_loop_guard_results.json` | Detecte le fallback repetitif `ACTION6 {"x":18,"y":0}`; interrompt apres 2 repetitions consecutives; switch legal `ACTION3`; support=0 |
| SAGE.2 - Known-game closed-loop policy probe | Fait | `theory/sage/known_game_policy_probe.py`, `tests/test_sage_known_game_policy_probe.py`, `diagnostics/sage/sage2_known_game_policy_probe_results.json` | Compare 4 policies sur le meme RESET bp35 (random legal, repeat ACTION6 fallback, SAGE.1 sans guard, SAGE.1b avec guard); metriques comparatives candidate-only; `outcome_status=SAGE_1B_IMPROVES_LOOP_DISCIPLINE_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.3 - Subgoal switch after success-like exhaustion | Fait | `theory/sage/subgoal_switcher.py`, `tests/test_sage_subgoal_switcher.py`, `diagnostics/sage/sage3_subgoal_switch_results.json` | Apres epuisement des cibles success-like, choisit un nouveau sous-but legal non repetitif parmi reposition / explore ACTION6 / contrefactuel actif / rerun M2-M3 / safe-hold; `outcome_status=SAGE_SWITCHES_TO_NEW_LEGAL_SUBGOAL_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.4 - Long-horizon transfer probe | Fait - gate echouee candidate-only | `theory/sage/long_horizon_transfer.py`, `tests/test_sage_long_horizon_transfer.py`, `diagnostics/sage/sage4_long_horizon_transfer_results.json` | Teste le transfert SAGE.3 vers ar25 sur budgets 50/150/300; 0/3 gates passed; repetition collapse, aucun subgoal switch; `outcome_status=SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.4b - Progress-stall trigger repair | Fait | `theory/sage/progress_stall_trigger.py`, `tests/test_sage_progress_stall_trigger.py`, `diagnostics/sage/sage4b_progress_stall_trigger_results.json` | Ajoute un trigger long-horizon generique sur repetition/stall; ar25 budget 150; `progress_stall_detected=true`, `subgoal_switches=49`, repetition post-trigger 0.41844, gate passed; support=0; aucun write A32/A33 |
| SAGE.4c - Long-horizon transfer rerun with progress-stall trigger | Fait | `theory/sage/long_horizon_progress_stall_transfer.py`, `tests/test_sage_long_horizon_progress_stall_transfer.py`, `diagnostics/sage/sage4c_long_horizon_progress_stall_transfer_results.json` | Relance ar25 budgets 50/150/300 avec trigger progress-stall; 3/3 gates passed; repetition reduite vs SAGE.4 sur tous les budgets; `outcome_status=SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.5 - First unknown-game bounded closed-loop probe | Fait | `theory/sage/unknown_game_bounded_probe.py`, `tests/test_sage_unknown_game_bounded_probe.py`, `diagnostics/sage/sage5_unknown_game_bounded_probe_results.json` | Premier probe public_unseen sur `sb26-7fbdac44`; budgets 50/150/300; 3/3 gates passed; unknown hygiene true; actions legales; repetition 0.0; terminal sous seuil 0.05; support=0; aucun write A32/A33 |
| SAGE.5b - Switch attribution and placeholder audit | Fait - dependance placeholder elevee | `theory/sage/switch_attribution_placeholder_audit.py`, `tests/test_sage_switch_attribution_placeholder_audit.py`, `diagnostics/sage/sage5b_switch_attribution_placeholder_audit.json` | Attribue 308 switches SAGE.5 : 306 success-like/loop-guard, 0 progress-stall, 2 terminal-guard; 297 placeholders `rerun_m2_m3`, 0 requests effectives; `outcome_status=SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY`; support=0; aucun write A32/A33 |
| SAGE.5c - Live mini-frontier generation from unknown state | Fait | `theory/sage/live_mini_frontier_generation.py`, `tests/test_sage_live_mini_frontier_generation.py`, `diagnostics/sage/sage5c_live_mini_frontier_results.json` | Convertit 20 placeholders `rerun_m2_m3` en 20 hypotheses + 20 requests M3 live-prefix candidate-only; effective_request_ratio=0.06734; residual placeholder ratio 0.905229; support=0; aucun write A32/A33 |
| SAGE.5d - Live-prefix mini-frontier M3 execution | Fait | `theory/sage/live_mini_frontier_m3_executor.py`, `tests/test_sage_live_mini_frontier_m3_executor.py`, `diagnostics/sage/sage5d_live_mini_frontier_m3_results.json` | Execute 8 requests SAGE.5c stratifiees en `LIVE_PREFIX_REPLAY_CONTEXT`; 8/8 replay exact + hash verifie; ACTION5/ACTION6 et object/local couverts; support_events=8, contradiction_events=0, mais support=0 et aucun write A32/A33 |
| SAGE.5e - Distributed live mini-frontier generation | Fait | `theory/sage/distributed_live_mini_frontier_generation.py`, `tests/test_sage_distributed_live_mini_frontier_generation.py`, `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json` | Distribue 8 requests par budget 50/150/300 au lieu de consommer tout le cap au budget 50; dedup par contexte/action/args/diff; execute 2 requests par budget; 6/6 replay exact + hash verifie; support_events=6, contradiction_events=0, mais support=0 et aucun write A32/A33 |
| SAGE.5f - Mini-frontier event consolidation | Fait | `theory/sage/mini_frontier_event_consolidation.py`, `tests/test_sage_mini_frontier_event_consolidation.py`, `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json` | Consolide les 6 events SAGE.5e en 3 clusters candidate-only; 2 clusters robustes multi-budget; 2 frontieres `ready_for_A32_review` non-verdict; support=0; aucun write A32/A33 |
| SAGE.5g - A32 review handoff compiler | Fait | `theory/sage/a32_review_handoff.py`, `tests/test_sage_a32_review_handoff.py`, `diagnostics/sage/sage5g_a32_review_handoff.json` | Compile 2 dossiers A32-review candidate-only et 4 followups; `ACTION6` requiert une diversite de controle; `ACTION5` requiert support + diversite de controle et remesure croisee du cluster lie non fusionne; support=0; aucune execution ni write A32/A33 |
| SAGE.5h - Controlled follow-up acquisition | Fait - partiel, surface de controle epuisee | `theory/sage/controlled_followup_acquisition.py`, `tests/test_sage_controlled_followup_acquisition.py`, `diagnostics/sage/sage5h_controlled_followup_acquisition.json` | Resout les 4 followups : 2 acquis, 2 bloques car seuls ACTION5/ACTION6 sont legaux; ACTION5 atteint 3 evenements comparables; remesure `local_patch` alignee mais `object_delta` divergente entre clusters 002/003; support=0; aucun write A32/A33 |
| SAGE.5i - Control-surface expansion | Fait - epuisement action-distinct borne | `theory/sage/control_surface_expansion.py`, `tests/test_sage_control_surface_expansion.py`, `diagnostics/sage/sage5i_control_surface_expansion.json` | Audite les 24 contextes candidats SAGE.5e en replay exact; tous exposent seulement ACTION5/ACTION6; aucune troisieme famille d'action; 7 options parametrees conservees sans requalification; proposition de decision protocolaire A32; support=0; aucun write A32/A33 |
| SAGE.5j - Pre-registered parameterized control acquisition | Fait - resultat mixte candidate-only | `theory/sage/parameterized_control_acquisition.py`, `tests/test_sage_parameterized_control_acquisition.py`, `diagnostics/sage/sage5j_parameterized_control_acquisition.json` | Execute exactement les 8 experiences A32.4 sans substitution; ACTION6 : 4/4 effets non discriminants (5 vs 5); ACTION5 : 4/4 effets discriminants (21 vs 4); 0 contradiction; 2 dossiers prets pour revue A32; support=0; aucun write A32/A33 |
| SAGE.6 - Second unknown-game bounded transfer | Fait - 3/3 gates passes sur wa30 | `theory/sage/second_unknown_game_transfer.py`, `tests/test_sage_second_unknown_game_transfer.py`, `diagnostics/sage/sage6_second_unknown_game_transfer_results.json` | Selectionne wa30 par ordre public_unseen fixe avant execution; exclut les jeux connus et sb26 source; budgets 50/150/300 passes; progress-stall detecte sur les 3 budgets; 98 switches; quarantaine A33.2 respectee; support=0; aucun write A32/A33 |
| SAGE.6a - Second-game switch attribution and live mini-frontier | Fait - 20 requests M3 candidate-only | `theory/sage/second_unknown_game_switch_frontier.py`, `tests/test_sage_second_unknown_game_switch_frontier.py`, `diagnostics/sage/sage6a_switch_attribution_mini_frontier.json` | Reproduit et attribue les 98 switches wa30 : 32 contrefactuels actifs, 34 repositionnements, 32 placeholders; distingue 1 garde terminale hors compteur source; convertit 20 placeholders en hypotheses + requests M3 live-prefix, reparties 4/12/4; support=0; aucun write A32/A33 |
| SAGE.6b - Stratified second-game M3 execution | Fait - 6/6 replays exacts | `theory/sage/second_unknown_game_m3_execution.py`, `tests/test_sage_second_unknown_game_m3_execution.py`, `diagnostics/sage/sage6b_second_unknown_game_m3_execution.json` | Selectionne 2 requests wa30 par budget avec 6 hashes distincts; execute ACTION2 contre ACTION1; 6/6 replay exact + hash verifie, 0 blocage; 5 effets bruts positifs et 1 neutre, mais support=0 et aucun write A32/A33 |
| SAGE.6c - Context-preserving event consolidation | Fait - motif +32 stable, exception neutre conservee | `theory/sage/second_unknown_game_event_consolidation.py`, `tests/test_sage_second_unknown_game_event_consolidation.py`, `diagnostics/sage/sage6c_second_unknown_game_event_consolidation.json` | Conserve 6 clusters singleton sans fusion; groupe 5 effets +32 sur les 3 budgets et 1 contexte neutre separe; frontiere compilable pour handoff mais non prete A32 car un seul controle distinct; support=0; aucun write A32/A33 |
| SAGE.6d - Second-game handoff compiler | Fait - 4 followups pre-enregistres | `theory/sage/second_unknown_game_handoff_compiler.py`, `tests/test_sage_second_unknown_game_handoff_compiler.py`, `diagnostics/sage/sage6d_second_unknown_game_handoff.json` | Compile 1 handoff candidate-only; pre-enregistre ACTION2/ACTION3 sur un contexte stable par budget et une replication exacte ACTION2/ACTION1 du contexte neutre; conserve les 6 frontieres de contexte; aucune execution, support=0, aucun write A32/A33 |
| SAGE.6e - Exact pre-registered followup execution | Fait - effet depend du controle candidate-only | `theory/sage/second_unknown_game_followup_execution.py`, `tests/test_sage_second_unknown_game_followup_execution.py`, `diagnostics/sage/sage6e_second_unknown_game_followup_execution.json` | Execute les 4 protocoles sans substitution; 4/4 replays exacts; ACTION2-ACTION3=0 sur les 3 budgets alors que ACTION2-ACTION1 valait +32; replication neutre ACTION2-ACTION1=0; 3 deviations pre-enregistrees, 1 condition satisfaite; support=0; aucun write A32/A33 |
| SAGE.6f - Control-dependence consolidation and A32 eligibility | Fait - 1 frontier A32 non-verdict eligible | `theory/sage/second_unknown_game_control_dependence_consolidation.py`, `tests/test_sage_second_unknown_game_control_dependence_consolidation.py`, `diagnostics/sage/sage6f_second_unknown_game_control_dependence_consolidation.json` | Consolide 10 observations dans 6 contextes non fusionnes; garde ACTION1/ACTION3 separes; 3 paires multi-budget avec gap stable 32; exception neutre repliquee; 14/14 criteres d'eligibilite; reformule le candidat comme contraste dependant du controle; support=0; aucun write A32/A33 |
| SAGE.7 - Third unknown-game bounded transfer | Fait - 3/3 gates passes, mini-frontiere parametree requise | `theory/sage/third_unknown_game_transfer.py`, `tests/test_sage_third_unknown_game_transfer.py`, `diagnostics/sage/sage7_third_unknown_game_transfer_results.json` | Selectionne tn36 par ordre public_unseen fixe apres exclusion de sb26/wa30; budgets 50/150/300 passes; 172 switches, 0 repetition d'argument; quarantaine A33.2/A33.3 respectee; surface live=11 variantes ACTION6 d'une seule famille; 145 placeholders M2/M3 non materialises; support=0; aucun write A32/A33 |

## SAGE.0 - Known-game closed-loop scaffold

Objectif :

- Verifier que la boucle SAGE minimale tient sur jeu connu, sans benchmark.
- Charger le contexte candidate-only M2.15, M3.7e, M3.7f et P1.
- Selectionner une action candidate depuis un contexte live.
- Executer un pas env dans un scaffold deterministe.
- Logger explicitement la frontiere offline-vs-live pour les contrefactuels.

Ajouts :

- `theory/sage/known_game_closed_loop_scaffold.py`
- `theory/sage/__init__.py`
- `tests/test_sage_known_game_closed_loop_scaffold.py`
- `diagnostics/sage/sage0_known_game_closed_loop_scaffold.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 2
- inputs read = `M2.15`, `M3.7e`, `M3.7f`, `P1`
- benchmark_run = false
- scaffold_only = true
- live_observations = 3
- env_steps_executed = 2
- actions selectionnees :
  - step 0 : `ACTION4`
  - step 1 : `ACTION6 {"x":12,"y":0}`
- m2_ready_for_m3 = 3
- m3_fused_requests_executed = 3
- blocked_capability_frontiers_logged = true
- offline_counterfactual_allowed = false
- active_counterfactual_collection_allowed = true
- env_state_restore_available = false
- offline_trace_context_is_observation_only = true
- live_env_context_authorizes_alternative_actions = true
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_0`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Logger contract :

```json
{
  "offline_counterfactual_allowed": false,
  "active_counterfactual_collection_allowed": true,
  "env_state_restore_available": false,
  "blocked_capability_frontiers_logged": true
}
```

Lecture :

SAGE.0 valide l'integration minimale : la boucle sait lire les hypotheses et
observations candidate-only, choisir une action candidate depuis un contexte
live, executer un pas env, puis produire un log scientifique disciplined. Il ne
s'agit pas d'un benchmark de performance et il n'y a aucun verdict A32/A33.

M3.7g offline counterfactual executor reste `NO-GO` tant qu'aucun etat env
complet/restorable n'est disponible. Les alternatives doivent etre collectees
activement en live.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_scaffold `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage0_known_game_closed_loop_scaffold.json `
  --budget 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_m3_offline_frame_counterfactual_probe.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.1 - Known-game technical live run

Objectif :

- Passer du scaffold SAGE.0 a un runner connu utilisant le vrai wrapper env.
- Verifier que les actions disponibles viennent de l'API live reelle.
- Selectionner uniquement des actions legales.
- Collecter une alternative contrefactuelle active par replay de prefixe live,
  jamais depuis une frame offline visuelle.
- Garder le run technique : pas de benchmark, pas de verdict, pas de support.

Ajouts :

- `theory/sage/known_game_closed_loop_runner.py`
- `theory/sage/live_prefix_counterfactual_collector.py`
- `tests/test_sage_known_game_closed_loop_runner.py`
- `tests/test_sage_live_prefix_counterfactual_collector.py`
- `diagnostics/sage/sage1_known_game_closed_loop_results.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 20
- technical_live_run = true
- benchmark_run = false
- inputs read = `M2.15`, `M3.7e`, `M3.7f`, `P1`
- available_actions_source = `real_env_live_api`
- real_env_available_actions_used = true
- synthetic_available_actions_used = false
- actions disponibles observees via env reel =
  `ACTION3`, `ACTION4`, `ACTION6`, `ACTION7`
- selected_action_always_legal = true
- invalid_action_selected = false
- live_steps_executed = 20
- env_actions = 20
- active_counterfactual_collection_attempted = true
- live_prefix_replay_exact = true
- live_prefix_replay_exact_count = 1
- counterfactual_collections = 1
- alternative collectee = `ACTION3`
- offline_counterfactual_allowed = false
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_1`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Details importants :

- Le point SAGE.0 a ete corrige : les actions disponibles de SAGE.1 ne viennent
  plus d'un scaffold permissif. Elles viennent du wrapper env bp35 reel.
- Le collecteur contrefactuel ne restaure pas un etat env depuis `grid_t`.
  Il rejoue `RESET -> prefix live`, verifie la signature, puis execute l'action
  alternative seulement si le replay est exact.
- Le premier replay prefix exact a collecte `ACTION3` apres le prefix
  `ACTION4`, avec changement d'etat observe.

Lecture :

SAGE.1 valide la capacite technique minimale pour une boucle connue reelle :
lecture des contextes candidate-only, action selection depuis API live,
execution env, et collecte contrefactuelle active par replay de prefixe. Ce run
ne prouve pas que SAGE joue bien, generalise ou decouvre une mecanique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_runner `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage1_known_game_closed_loop_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.1b - Known-game policy loop guard

Objectif :

- Corriger le signal negatif observe dans SAGE.1 : apres epuisement des cibles
  success-like, le fallback repetait `ACTION6 {"x":18,"y":0}`.
- Detecter les repetitions consecutives du meme couple `(action, args)`.
- Interrompre le fallback cible apres un seuil court.
- Selectionner un switch legal depuis `real_env_live_api`, en priorite la
  frontiere active `ACTION3`.
- Garder le statut technique : pas de benchmark, pas de verdict, pas de support.

Ajouts :

- `theory/sage/policy_loop_guard.py`
- integration optionnelle dans `theory/sage/known_game_closed_loop_runner.py`
- `tests/test_sage_policy_loop_guard.py`
- `diagnostics/sage/sage1b_policy_loop_guard_results.json`

Run du 2026-06-29 :

- jeu = `bp35-0a0ad940`
- budget = 20
- technical_live_run = true
- benchmark_run = false
- loop_guard_enabled = true
- loop_guard_max_repeats = 2
- repeated_same_action_args_detected = true
- fallback_loop_interrupted = true
- switch_action_selected_after_exhaustion = true
- loop_guard_switches = 4
- max_same_action_arg_repeats = 2
- blocked fallback = `ACTION6 {"x":18,"y":0}`
- switch action = `ACTION3`
- selected_action_always_legal = true
- invalid_action_selected = false
- synthetic_available_actions_used = false
- policy_result_counted_as_confirmation = false
- support = 0
- truth_status = `NOT_EVALUATED_BY_SAGE_1B`
- revision_status = `CANDIDATE_ONLY`
- a32_write_performed = false
- a33_write_performed = false

Lecture :

SAGE.1b ne rend pas la policy performante. Il retire seulement un attracteur
evident du runner connu : le meme fallback `ACTION6 {"x":18,"y":0}` ne peut
plus etre choisi indefiniment. Le switch reste un choix technique legal, pas une
confirmation scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_closed_loop_runner `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage1b_policy_loop_guard_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940 `
  --enable-loop-guard `
  --loop-guard-max-repeats 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

## SAGE.2 - Known-game closed-loop policy probe

Objectif :

- Passer de "le runner tourne legalement" (SAGE.1/1b) a une premiere mesure
  *comparative* sur jeu connu : SAGE produit-il une trajectoire meilleure qu'un
  baseline simple, sous budget court/moyen ?
- Rejouer le meme RESET bp35 reel pour chaque policy et comparer des metriques
  comportementales, sans benchmark de score humain et sans verdict scientifique.
- Garder la discipline : `support=0`, `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_2`, aucun write A32/A33.

Ajouts :

- `theory/sage/known_game_policy_probe.py`
- `tests/test_sage_known_game_policy_probe.py`
- `diagnostics/sage/sage2_known_game_policy_probe_results.json`

Baselines comparees (toutes depuis le meme RESET, API live reelle) :

- `random_legal` - action legale aleatoire (deterministe par seed)
- `repeat_action6_fallback` - repete le fallback cible `ACTION6`
- `sage1_without_loop_guard` - selecteur SAGE.1 sans loop guard
- `sage1b_with_loop_guard` - selecteur SAGE.1b avec loop guard

Metriques par policy :

- `levels_completed`
- `terminal_rate`
- `state_changed_rate`
- `unique_state_signatures`
- `repeated_action_arg_rate`
- `max_same_action_arg_repeats`
- `loop_guard_switches`
- `active_counterfactual_collections` (collecte active live, SAGE seulement)
- `policy_result_counted_as_confirmation = false`
- `support = 0`

Run du 2026-06-30 (jeu `bp35-0a0ad940`, budget 20, seed 0, guard max repeats 2) :

| policy | levels | terminal_rate | state_changed_rate | unique_sigs | repeated_arg_rate | max_repeats | loop_guard_switches | counterfactuals |
|---|---|---|---|---|---|---|---|---|
| random_legal | 0 | 0.0 | 1.0 | 21 | 0.20 | 2 | 0 | 0 |
| repeat_action6_fallback | 0 | 0.0 | 1.0 | 21 | 0.95 | 20 | 0 | 0 |
| sage1_without_loop_guard | 0 | 0.0 | 1.0 | 21 | 0.60 | 13 | 0 | 1 |
| sage1b_with_loop_guard | 0 | 0.0 | 1.0 | 21 | 0.20 | 2 | 4 | 1 |

- `sage1b_loop_discipline_improved_vs_sage1 = true`
- `sage1b_explores_at_least_as_much_as_repeat_fallback = true`
- `outcome_status = SAGE_1B_IMPROVES_LOOP_DISCIPLINE_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- champ nomme `outcome_status` (jamais `verdict`) pour eviter toute ambiguite avec A32/A33
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_2`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

Sur bp35, chaque action change l'etat (`state_changed_rate = 1.0` partout), donc
le discriminant n'est pas l'exploration brute mais la *discipline de boucle*. Le
loop guard fait chuter `repeated_action_arg_rate` de 0.95 (repeat) et 0.60
(SAGE.1) a 0.20, et plafonne `max_same_action_arg_repeats` a 2, tout en restant
au moins aussi exploratoire que le repeat fallback. SAGE.1b est donc le seul des
quatre a combiner faible repetition ET collecte contrefactuelle active.

C'est un premier signal comportemental candidate-only, pas une preuve de
competence agentique : aucun policy ne progresse de niveau (`levels_completed = 0`),
le switch reste technique, et le probe ne convertit aucun resultat en support
scientifique. L'`outcome_status` est candidate-only et n'ecrit ni A32 ni A33.

Limites encore ouvertes (non resolues par SAGE.2) :

- Apres epuisement des cibles success-like, le switch ne choisit pas encore un
  nouveau sous-but intelligent (repositionnement, contrefactuel cible, relance
  M2/M3, mode exploration).
- Jeu inconnu : pas encore evalue.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.known_game_policy_probe `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage2_known_game_policy_probe_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940 `
  --seed 0 `
  --loop-guard-max-repeats 2
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Prochaine etape : SAGE.3 (ci-dessous) - apres epuisement des cibles
success-like, choisir un nouveau sous-but au lieu de seulement eviter la boucle.

## SAGE.3 - Subgoal switch after success-like exhaustion

Objectif :

- SAGE.1b retire l'attracteur degenere, mais apres epuisement des cibles
  ACTION6 success-like (ou declenchement du guard) SAGE ne sait pas encore
  *quoi chercher ensuite*. SAGE.3 est un module de transition de mode, pas un
  meilleur guard.
- Quand l'epuisement/loop est detecte, choisir un nouveau sous-but legal et non
  repetitif parmi 5 modes :
  1. `active_counterfactual_collection` - collecter un contrefactuel live
  2. `repositioning` - repositionner via l'action de repositionnement
  3. `explore_new_candidate_action6_target` - essayer une cible ACTION6 inutilisee
  4. `rerun_m2_m3` - flag de re-derivation + action legale neutre
  5. `stop_safe_hold` - tenir si un risque terminal est present
- Critere minimal : PAS gagner, mais produire apres epuisement un comportement
  nouveau, legal, non trivial, non repetitif et mesurable.
- Discipline : `support=0`, `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_3`, champ `outcome_status` (jamais
  `verdict`), aucun write A32/A33.

Ajouts :

- `theory/sage/subgoal_switcher.py`
- `tests/test_sage_subgoal_switcher.py`
- `diagnostics/sage/sage3_subgoal_switch_results.json`

Detection du declencheur :

- L'epuisement est detecte quand `select_runner_action` (sans loop guard) tombe
  sur `decision_reason == candidate_policy_live_target_fallback`.
- `trigger_reason = success_like_targets_exhausted_or_loop_guard` (canonique).
- La rotation round-robin sur les modes *faisables* garantit la diversite et la
  non-repetition; chaque action candidate est verifiee != action bloquee et !=
  action precedente.

Contrat (par evenement de switch) :

```json
{
  "subgoal_switch_triggered": true,
  "trigger_reason": "success_like_targets_exhausted_or_loop_guard",
  "selected_subgoal": "active_counterfactual_collection_or_repositioning",
  "selected_action_legal": true,
  "policy_result_counted_as_confirmation": false,
  "support": 0,
  "truth_status": "NOT_EVALUATED_BY_SAGE_3",
  "a32_write_performed": false,
  "a33_write_performed": false
}
```

Metriques SAGE.3 (en plus des metriques SAGE.2) :

- `subgoal_switches`
- `subgoal_switch_success_rate` (legal ET state_changed ET non repetitif)
- `subgoals_used`
- `new_candidate_targets_discovered`
- `active_counterfactuals_after_exhaustion`
- `post_switch_state_changed_rate`
- `post_switch_repeat_rate`
- `post_switch_terminal_rate`
- `levels_completed`

Run du 2026-06-30 (jeu `bp35-0a0ad940`, budget 20) :

- `env_steps = 20`
- `subgoal_switches = 13`
- `subgoal_switch_success_rate = 1.0`
- `subgoals_used = [active_counterfactual_collection, repositioning, rerun_m2_m3]`
- `new_candidate_targets_discovered = 0`
- `active_counterfactuals_after_exhaustion = 4`
- `post_switch_state_changed_rate = 1.0`
- `post_switch_repeat_rate = 0.0`
- `post_switch_terminal_rate = 0.0`
- `placeholder_actions_used = 4`
- `placeholder_action_counted_as_subgoal_success = false`
- `rerun_m2_m3_requested = 4`
- `rerun_m2_m3_effective_requests_generated = 0`
- `levels_completed = 0`
- `selected_action_always_legal = true`
- `outcome_status = SAGE_SWITCHES_TO_NEW_LEGAL_SUBGOAL_CANDIDATE_ONLY`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_3`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

Apres epuisement, SAGE ne repete plus : il alterne reposition, contrefactuel
actif et rerun M2-M3, chaque switch produisant un changement d'etat sans
repetition consecutive (`post_switch_repeat_rate = 0.0`,
`subgoal_switch_success_rate = 1.0`) et sans entree terminale. Le critere minimal
SAGE.3 est atteint : un comportement nouveau, legal, non trivial et mesurable
apres epuisement.

Limite honnete observee sur bp35 : `explore_new_candidate_action6_target` n'a pas
ete faisable aux frames d'epuisement (l'unique cible ACTION6 live par frame
etait deja utilisee), donc `new_candidate_targets_discovered = 0`. Le module
degrade proprement vers les 3 autres modes au lieu d'inventer une cible. Aucun
niveau n'est franchi (`levels_completed = 0`) : SAGE sait desormais reformuler
un sous-but, mais ce n'est pas encore une competence de jeu.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.subgoal_switcher `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage3_subgoal_switch_results.json `
  --budget 20 `
  --game-id bp35-0a0ad940
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4 a maintenant ete execute sur ar25 connu. Le resultat ci-dessous remplace
l'ancien ordre conseille : le transfert long-horizon echoue proprement, donc le
jeu inconnu reste bloque tant que le trigger de switch ne transfere pas au moins
partiellement.

## SAGE.4 - Long-horizon transfer probe

Objectif :

- Tester si la discipline de boucle apprise sur le cas court bp35
  (SAGE.1b + SAGE.3) se transfere a un jeu connu plus long-horizon :
  `ar25-e3c63847`.
- Evaluer plusieurs budgets (`50`, `150`, `300`) sans demander de resoudre ar25.
- Gate minimal : actions toujours legales, pas de collapse repetitif,
  sous-buts effectivement declenches, et au moins une capacite de sous-but utile.
- Garder le statut candidate-only : `support=0`,
  `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_4`, aucun write A32/A33.

Ajouts :

- `theory/sage/long_horizon_transfer.py`
- `tests/test_sage_long_horizon_transfer.py`
- `diagnostics/sage/sage4_long_horizon_transfer_results.json`

Gate SAGE.4 :

```json
{
  "selected_action_always_legal": true,
  "no_repeat_collapse": true,
  "post_switch_repeat_discipline": true,
  "subgoal_switches_happened": true,
  "useful_subgoal_capability": true,
  "gate_passed": true
}
```

Run du 2026-06-30 (jeu `ar25-e3c63847`, budgets `50`, `150`, `300`) :

| budget | gate_passed | repeated_arg_rate | unique_sigs | subgoal_switches | subgoal_success | active_counterfactuals | new_targets | levels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | false | 0.96 | 7 | 0 | 0.0 | 0 | 0 | 0 |
| 150 | false | 0.986667 | 7 | 0 | 0.0 | 0 | 0 | 0 |
| 300 | false | 0.993333 | 7 | 0 | 0.0 | 0 | 0 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 0`
- `budgets_total = 3`
- `any_budget_gate_passed = false`
- `all_budgets_gate_passed = false`
- `discovered_new_candidate_targets_beyond_bp35 = false`
- `outcome_status = SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4 est un resultat negatif utile. La legalite tient
(`selected_action_always_legal = true` sur tous les budgets), mais la discipline
de boucle ne transfere pas a ar25 : le taux de repetition monte de `0.96` a
`0.993333`, aucun switch de sous-but n'est declenche, aucune collecte
contrefactuelle active n'a lieu, et aucune nouvelle cible candidate n'est
decouverte. Les gates echouent donc sur `no_repeat_collapse`,
`subgoal_switches_happened` et `useful_subgoal_capability`.

Interpretation :

SAGE.3 est valide localement sur bp35, mais son trigger d'epuisement ne se
generalise pas tel quel a ar25. SAGE.4 ne dit pas que SAGE est faux ou inutile ;
il dit que le mecanisme actuel de switch est trop attache a la dynamique bp35 et
ne produit pas encore une strategie long-horizon transferable. Ce resultat ne
devient pas support scientifique et ne modifie aucune hypothese A32/A33.

Point technique :

L'artefact SAGE.4 n'a pas de champ `summary` top-level ; le resume utile est
dans `comparison`, et les mesures detaillees sont dans `per_budget_results`.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.long_horizon_transfer `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage4_long_horizon_transfer_results.json `
  --game-id ar25-e3c63847 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4b a maintenant ete execute : le trigger long-horizon repare le defaut
principal de SAGE.4 en declenchant des switches sur ar25 sans attendre le
fallback bp35-style.

## SAGE.4b - Progress-stall trigger repair

Objectif :

- Ajouter un declencheur generique de stall long-horizon, independant de
  `decision_reason == candidate_policy_live_target_fallback`.
- Declencher le switch sur repetition/stall observable :
  - meme couple `(action, args)` repete au moins `K`
  - pas de progression de niveau dans la fenetre
  - faible nouveaute d'etats ou taux de repetition local eleve
- Verifier sur ar25 que le switcher est appele, que la repetition baisse, et que
  les actions restent legales.
- Garder le statut candidate-only : `support=0`, aucun verdict, aucun write
  A32/A33.

Ajouts :

- `theory/sage/progress_stall_trigger.py`
- integration optionnelle dans `theory/sage/subgoal_switcher.py`
- `tests/test_sage_progress_stall_trigger.py`
- `diagnostics/sage/sage4b_progress_stall_trigger_results.json`

Trigger :

```json
{
  "trigger_reason": "progress_stall_or_repeat_collapse",
  "same_action_args_repeated": true,
  "low_state_novelty": true,
  "no_level_progress": true,
  "switch_required": true,
  "support": 0
}
```

Gate SAGE.4b :

- `selected_action_always_legal = true`
- `progress_stall_detected = true`
- `subgoal_switches > 0`
- `repeat_collapse_interrupted = true`
- `repeated_action_arg_rate_after_trigger < repeated_action_arg_rate_before_trigger`
- `repeated_action_arg_rate_after_trigger < 0.9`
- `active_counterfactual_collection_attempted OR rerun_m2_m3_requested`
- `support = 0`
- aucun write A32/A33

Run du 2026-06-30 (jeu `ar25-e3c63847`, budget 150) :

- `progress_stall_trigger_enabled = true`
- `progress_stall_detected = true`
- `first_progress_stall_trigger_step = 4`
- `subgoal_switches = 49`
- `progress_stall_switches = 49`
- `repeat_collapse_interrupted = true`
- `repeated_action_arg_rate_before_trigger = 0.5`
- `repeated_action_arg_rate_after_trigger = 0.41844`
- `post_trigger_repeat_failure_threshold = 0.9`
- `active_counterfactual_collection_attempted = true`
- `active_counterfactuals_after_exhaustion = 8`
- `rerun_m2_m3_requested = 9`
- `rerun_m2_m3_effective_requests_generated = 0`
- `new_candidate_targets_discovered = 0`
- `unique_state_signatures = 65`
- `levels_completed = 0`
- `selected_action_always_legal = true`
- `invalid_action_selected = false`
- `gate_passed = true`
- `outcome_status = SAGE_PROGRESS_STALL_TRIGGER_REPAIRED_CANDIDATE_ONLY`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4B`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4b repare le verrou precis observe dans SAGE.4 : le switcher est maintenant
appele sur ar25 meme sans le signal bp35 `candidate_policy_live_target_fallback`.
Le collapse repetitif est interrompu, la legalite est preservee, et les sous-buts
`repositioning`, `active_counterfactual_collection` et `rerun_m2_m3` sont utilises
en contexte long-horizon.

Ce n'est toujours pas une competence de jeu : `levels_completed = 0`, aucune
nouvelle cible candidate n'est decouverte, et `rerun_m2_m3` reste un placeholder
non effectif (`rerun_m2_m3_effective_requests_generated = 0`). Le resultat est
un gate de capacite candidate-only, pas une preuve de performance.

Point technique :

Le runner stoppe maintenant proprement en `stop_safe_hold` lorsqu'un etat terminal
est atteint, au lieu d'essayer une action live apres `GAME_OVER`.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.progress_stall_trigger `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --out diagnostics\sage\sage4b_progress_stall_trigger_results.json `
  --game-id ar25-e3c63847 `
  --budget 150 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.4c a maintenant ete execute. Le trigger progress-stall ne se contente plus
de passer un smoke test a budget 150 : il repare le transfert long-horizon sur
les trois budgets ar25 testes, en restant candidate-only.

## SAGE.4c - Long-horizon transfer rerun with progress-stall trigger

Objectif :

- Rejouer le protocole SAGE.4 sur `ar25-e3c63847` avec le trigger
  progress-stall active.
- Evaluer les budgets `50`, `150` et `300` contre le baseline SAGE.4.
- Verifier que la repetition baisse nettement, que les switches de sous-but se
  declenchent, et que les actions restent legales.
- Garder le statut candidate-only : `support=0`,
  `policy_result_counted_as_confirmation=false`,
  `truth_status=NOT_EVALUATED_BY_SAGE_4C`, aucun write A32/A33.

Ajouts :

- `theory/sage/long_horizon_progress_stall_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_long_horizon_progress_stall_transfer.py`
- `diagnostics/sage/sage4c_long_horizon_progress_stall_transfer_results.json`

Gate SAGE.4c :

```json
{
  "selected_action_always_legal": true,
  "progress_stall_detected": true,
  "subgoal_switches_happened": true,
  "no_repeat_collapse": true,
  "post_switch_repeat_discipline": true,
  "repeated_action_arg_rate_lower_than_sage4": true,
  "active_counterfactual_or_rerun_requested": true,
  "support": 0
}
```

Run du 2026-06-30 (jeu `ar25-e3c63847`, budgets `50`, `150`, `300`) :

| budget | gate_passed | repeated_arg_rate | SAGE4_repeat | subgoal_switches | active_counterfactuals | rerun_m2_m3 | unique_sigs | terminal_rate | levels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | true | 0.44 | 0.96 | 15 | 5 | 3 | 25 | 0.0 | 0 |
| 150 | true | 0.42069 | 0.986667 | 49 | 8 | 9 | 65 | 0.006897 | 0 |
| 300 | true | 0.42069 | 0.993333 | 49 | 8 | 9 | 65 | 0.006897 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 3`
- `budgets_total = 3`
- `any_budget_gate_passed = true`
- `all_budgets_gate_passed = true`
- `budgets_with_progress_stall_detected = [50, 150, 300]`
- `budgets_with_repetition_improved_vs_sage4 = [50, 150, 300]`
- `budgets_with_subgoal_switches = [50, 150, 300]`
- `baseline_sage4_outcome_status = SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY`
- `outcome_status = SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY`
- `outcome_status_is_candidate_only = true`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_4C`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.4c ferme le verrou ouvert par SAGE.4 : avec le trigger progress-stall, le
transfert de discipline de boucle fonctionne sur tous les budgets ar25 testes.
Le collapse repetitif de SAGE.4 est reduit de `0.96/0.986667/0.993333` a
`0.44/0.42069/0.42069`, les switches se declenchent, et des contrefactuels
actifs sont collectes.

Ce n'est toujours pas une competence de jeu ni un benchmark de performance :
`levels_completed = 0` sur tous les budgets, `new_candidate_targets_discovered = 0`,
et `rerun_m2_m3_effective_requests_generated = 0`. Le taux terminal non nul sur
les budgets `150` et `300` (`0.006897`) doit rester surveille. SAGE.4c valide
une capacite comportementale candidate-only : detecter un stall long-horizon et
sortir de la repetition, pas resoudre ar25.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.long_horizon_progress_stall_transfer `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --baseline-sage4 diagnostics\sage\sage4_long_horizon_transfer_results.json `
  --out diagnostics\sage\sage4c_long_horizon_progress_stall_transfer_results.json `
  --game-id ar25-e3c63847 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5 a maintenant ete execute sur un seul jeu public_unseen. Le resultat
ci-dessous remplace le plan prospectif : le probe inconnu borne passe ses gates,
mais reste strictement candidate-only et ne valide pas une competence de jeu.

## SAGE.5 - First unknown-game bounded closed-loop probe

Objectif :

- Executer une premiere boucle SAGE sur un jeu inconnu du point de vue M2.14 :
  `sb26-7fbdac44`.
- Garder le run borne : budgets `50`, `150`, `300`, sans objectif de benchmark.
- Comparer des baselines minimales :
  - `random_legal`
  - `neutral_legal_fallback`
  - contexte source `SAGE.4c`
- Verifier seulement les proprietes techniques :
  - hygiene unknown-game
  - actions legales
  - detecteur progress-stall actif
  - au moins un switch de sous-but sur au moins un budget
  - pas de collapse repetitif catastrophique
  - garde terminal sous seuil
  - `support=0`, aucun write A32/A33

Ajouts :

- `theory/sage/unknown_game_bounded_probe.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_unknown_game_bounded_probe.py`
- `diagnostics/sage/sage5_unknown_game_bounded_probe_results.json`

Gate SAGE.5 :

```json
{
  "unknown_game": true,
  "no_human_trace_for_game": true,
  "no_game_specific_prior": true,
  "selected_action_always_legal": true,
  "progress_stall_detector_runs": true,
  "subgoal_switches_happened": true,
  "no_catastrophic_repeat_collapse": true,
  "terminal_rate_guarded": true,
  "terminal_rate_under_threshold": true,
  "offline_counterfactual_allowed": false,
  "support": 0
}
```

Run du 2026-07-05 :

- `game_id = sb26-7fbdac44`
- `short_game_id = sb26`
- `split = unseen`
- `unknown_game = true`
- `no_human_trace_for_game = true`
- `no_m2_arc_lewm_trace_for_game = true`
- `no_game_specific_prior = true`
- `terminal_rate_threshold = 0.05`
- `source_sage4c_all_budgets_gate_passed = true`

Resultats SAGE.5 :

| budget | gate_passed | switches | progress_stall_detected | repeated_arg_rate | terminal_rate | active_cf | rerun_m2_m3 | new_targets | unique_sigs | levels |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | true | 50 | false | 0.0 | 0.0 | 0 | 47 | 3 | 51 | 0 |
| 150 | true | 128 | false | 0.0 | 0.007812 | 0 | 125 | 3 | 129 | 0 |
| 300 | true | 128 | false | 0.0 | 0.007812 | 0 | 125 | 3 | 129 | 0 |

Baselines observees :

| budget | baseline | repeated_arg_rate | terminal_rate | legal | unique_sigs | levels |
|---|---|---:|---:|---|---:|---:|
| 50 | neutral_legal_fallback | 0.98 | 0.0 | true | 51 | 0 |
| 50 | random_legal | 0.14 | 0.0 | true | 51 | 0 |
| 150 | neutral_legal_fallback | 0.984375 | 0.015625 | true | 65 | 0 |
| 150 | random_legal | 0.119658 | 0.008547 | true | 118 | 0 |
| 300 | neutral_legal_fallback | 0.984375 | 0.015625 | true | 65 | 0 |
| 300 | random_legal | 0.119658 | 0.008547 | true | 118 | 0 |

Comparaison :

- `budgets_evaluated = [50, 150, 300]`
- `budgets_gate_passed = 3`
- `budgets_total = 3`
- `all_budgets_gate_passed = true`
- `any_budget_gate_passed = true`
- `budgets_with_subgoal_switches = [50, 150, 300]`
- `budgets_without_catastrophic_repeat_collapse = [50, 150, 300]`
- `budgets_terminal_rate_under_threshold = [50, 150, 300]`
- `budgets_with_progress_stall_detected = []`
- `progress_stall_detector_runs = true`
- `offline_counterfactual_allowed = false`
- `active_counterfactual_collection_allowed = true`
- `outcome_status = SAGE_UNKNOWN_BOUNDED_PROBE_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.5 valide le premier passage technique sur jeu inconnu borne : le jeu est
bien hors traces M2.14/human traces, les actions restent legales, les switches
de sous-but se produisent sur les trois budgets, la repetition tombe a `0.0`,
et le taux terminal reste sous le seuil `0.05`.

Point important : le detecteur progress-stall etait actif, mais il n'a pas ete
declenche (`budgets_with_progress_stall_detected = []`). Sur `sb26`, le chemin
`success_like_targets_exhausted_or_loop_guard` suffisait a produire des
switches. SAGE.5 valide donc une boucle inconnue bornee et disciplined, pas une
generalisation du trigger progress-stall lui-meme.

Limites :

- `levels_completed = 0` sur tous les budgets.
- `active_counterfactuals_after_exhaustion = 0`.
- `rerun_m2_m3_effective_requests_generated = 0`.
- Les `new_candidate_targets_discovered = 3` sont des cibles candidates live,
  pas des preuves scientifiques.
- Le resultat ne confirme aucune mecanique et n'autorise aucun write A32/A33.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.unknown_game_bounded_probe `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage4c diagnostics\sage\sage4c_long_horizon_progress_stall_transfer_results.json `
  --m2-dataset-manifest diagnostics\m2\arc_lewm_dataset_manifest.json `
  --human-traces-dir human_traces `
  --out diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --game-id sb26-7fbdac44 `
  --budgets 50 150 300 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --terminal-rate-threshold 0.05
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5b a maintenant ete execute. Il confirme que SAGE.5 est une boucle inconnue
safe et non repetitive, mais que la majorite des switches reste un placeholder
`rerun_m2_m3` non effectif.

## SAGE.5b - Switch attribution and placeholder audit

Objectif :

- Attribuer chaque switch SAGE.5 a une cause explicite :
  - `success_like_targets_exhausted_or_loop_guard`
  - `progress_stall_or_repeat_collapse`
  - `terminal_guard`
- Mesurer les switches qui produisent une action exploratoire reelle.
- Mesurer les switches qui tombent sur le placeholder `rerun_m2_m3`.
- Calculer `rerun_m2_m3_effective_requests_generated / rerun_m2_m3_requested`.
- Garder le statut candidate-only : `support=0`, aucun verdict, aucun write
  A32/A33.

Ajouts :

- `theory/sage/switch_attribution_placeholder_audit.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_switch_attribution_placeholder_audit.py`
- `diagnostics/sage/sage5b_switch_attribution_placeholder_audit.json`

Run du 2026-07-05 :

| budget | switches | success-like / loop-guard | progress-stall | terminal-guard | exploratory | placeholder | placeholder_ratio | rerun_requested | effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50 | 50 | 0 | 0 | 3 | 47 | 0.94 | 47 | 0 |
| 150 | 129 | 128 | 0 | 1 | 3 | 125 | 0.968992 | 125 | 0 |
| 300 | 129 | 128 | 0 | 1 | 3 | 125 | 0.968992 | 125 | 0 |

Totals :

- `switches_total = 308`
- `switches_due_to_success_like_targets_exhausted_or_loop_guard = 306`
- `switches_due_to_progress_stall_or_repeat_collapse = 0`
- `switches_due_to_terminal_guard = 2`
- `true_exploratory_switches = 9`
- `active_counterfactual_switches_total = 0`
- `active_counterfactuals_after_exhaustion_total = 0`
- `placeholder_rerun_m2_m3_switches = 297`
- `placeholder_switch_ratio = 0.964286`
- `placeholder_dependency_threshold = 0.5`
- `placeholder_dependency_under_threshold = false`
- `rerun_m2_m3_requested = 297`
- `rerun_m2_m3_effective_requests_generated = 0`
- `effective_request_ratio = 0.0`
- `levels_completed_max = 0`
- `outcome_status = SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5B`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Lecture :

SAGE.5b confirme le point subtil de SAGE.5 : le detecteur progress-stall tourne,
mais ne cause aucun switch sur `sb26`. Les switches viennent presque tous du
chemin `success_like_targets_exhausted_or_loop_guard`, avec deux safe holds lies
au terminal guard. La boucle reste disciplined, mais elle n'est pas encore
apprenante : `rerun_m2_m3` est demande 297 fois et ne genere aucune request
effective.

Les 9 switches exploratoires correspondent aux nouvelles cibles candidates
locales, mais le reste du comportement est domine par le placeholder. SAGE.5b
n'est donc pas un gate de progression ; c'est une fermeture propre du diagnostic
placeholder.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.switch_attribution_placeholder_audit `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --out diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --placeholder-dependency-threshold 0.5
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5c a maintenant ete execute. Il transforme une partie du placeholder
`rerun_m2_m3` en hypotheses et requests M3 candidate-only effectives, sans
pretendre que ces requests sont deja support ou evidence.

## SAGE.5c - Live mini-frontier generation from unknown state

Objectif :

- Remplacer une partie des placeholders `rerun_m2_m3` par une mini-frontier live.
- Capturer `frame_before`, action, `frame_after` par replay prefix live.
- Calculer un diff deterministe :
  - cellules changees
  - bbox du changement
  - transitions de couleurs
  - couleurs ajoutees/supprimees
  - composantes creees/supprimees
  - terminal/non-terminal
- Generer une hypothese candidate simple :
  - `local_patch_change_candidate`
  - `object_delta_candidate`
  - `terminal_risk_candidate`
  - `no_effect_or_stale_candidate`
- Compiler une request M3 minimale en `LIVE_PREFIX_REPLAY_CONTEXT` quand un
  controle legal est disponible.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_live_mini_frontier_generation.py`
- `diagnostics/sage/sage5c_live_mini_frontier_results.json`

Run du 2026-07-05 :

- `game_id = sb26-7fbdac44`
- budgets = `[50, 150, 300]`
- `max_generated_requests = 20`
- `rerun_m2_m3_requested = 297`
- `effective_requests_generated = 20`
- `effective_request_ratio = 0.06734`
- `mini_frontier_hypotheses_generated = 20`
- `mini_frontier_m3_requests = 20`
- `unresolved_placeholder_switches_after_generation = 277`
- `source_placeholder_switch_ratio = 0.970588`
- `residual_placeholder_switch_ratio = 0.905229`
- `true_exploratory_or_scientific_switches = 29`
- `levels_completed_max = 0`
- `outcome_status = SAGE_LIVE_MINI_FRONTIER_GENERATED_CANDIDATE_ONLY`
- `generated_requests_counted_as_support = false`
- `mini_frontier_counted_as_evidence = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5C`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Par budget :

| budget | placeholders | generated | effective_ratio | source_placeholder_ratio | residual_placeholder_ratio | exploratory_or_scientific | generation_budget_exhausted |
|---|---:|---:|---:|---:|---:|---:|---|
| 50 | 47 | 20 | 0.425532 | 0.94 | 0.54 | 23 | true |
| 150 | 125 | 0 | 0.0 | 0.976562 | 0.976562 | 3 | true |
| 300 | 125 | 0 | 0.0 | 0.976562 | 0.976562 | 3 | true |

Familles generees :

- `local_patch_change_candidate = 19`
- `object_delta_candidate = 1`

Exemple de request generee :

- `request_id = m2m3::sage5c::live_mini_frontier::050::0001`
- `game_id = sb26-7fbdac44`
- `context_state_origin = sage5_live_prefix_frame_before`
- `replayability = LIVE_PREFIX_REPLAY_CONTEXT`
- `target_action = ACTION5`
- `metric = local_patch_before_after`
- `status = READY_FOR_M3`
- `support = 0`

Lecture :

SAGE.5c est le premier pas ou le placeholder `rerun_m2_m3` devient partiellement
effectif : 20 transitions live inconnues produisent des hypotheses et requests M3
chargeables, au lieu de rester de simples demandes non executees. Le ratio
`20/297 = 0.06734` est encore faible, mais il suffit a transformer la boucle de
survie disciplined en debut de boucle d'apprentissage active.

Limites :

- La generation est capee a 20 requests, donc 277 placeholders restent non
  transformes.
- Les budgets 150/300 ne generent rien apres epuisement du cap global.
- Les requests ne sont pas encore executees par M3.
- `levels_completed_max = 0`.
- Les hypotheses generees ne comptent pas comme support/evidence.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.live_mini_frontier_generation `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-sage5b diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --out diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75 `
  --max-generated-requests 20
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5d a maintenant ete execute. Il ferme la chaine unknown-game minimale :
jeu inconnu -> placeholder SAGE -> mini-frontier SAGE.5c -> request M3
`LIVE_PREFIX_REPLAY_CONTEXT` -> replay exact -> mesure candidate-only.

## SAGE.5d - Live-prefix mini-frontier M3 execution

Objectif :

- Lire `diagnostics/sage/sage5c_live_mini_frontier_results.json`.
- Selectionner 4 a 8 requests SAGE.5c stratifiees :
  - au moins une `object_delta_candidate`
  - plusieurs `local_patch_change_candidate`
  - au moins une `target_action=ACTION5`
  - au moins une `target_action=ACTION6`
- Rejouer le prefixe live exact depuis RESET.
- Verifier `context_snapshot_hash` avant toute action cible ou controle.
- Executer l'action target dans un env replaye.
- Executer le controle dynamique dans un env replaye separe.
- Mesurer `local_patch_before_after`.
- Produire seulement des observations candidate-only :
  `support_events`, `contradiction_events`, `blocked_replay_events`,
  `execution_failures`.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/live_mini_frontier_m3_executor.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_live_mini_frontier_m3_executor.py`
- `diagnostics/sage/sage5d_live_mini_frontier_m3_results.json`

Run du 2026-07-05 :

- source = `diagnostics/sage/sage5c_live_mini_frontier_results.json`
- `source_sage5c_effective_requests_generated = 20`
- `selected_requests = 8`
- `requests_executed = 8`
- `requests_blocked = 0`
- `live_prefix_replay_exact_events = 8`
- `context_snapshot_hash_verified_events = 8`
- `families_executed = {"object_delta_candidate":1, "local_patch_change_candidate":7}`
- `target_actions_executed = {"ACTION5":5, "ACTION6":3}`
- `support_events = 8`
- `contradiction_events = 0`
- `neutral_events = 0`
- `blocked_replay_events = 0`
- `execution_failures = 0`
- `outcome_status = SAGE_LIVE_MINI_FRONTIER_M3_EXECUTED_CANDIDATE_ONLY`
- `support_events_counted_as_support = false`
- `mini_frontier_execution_counted_as_evidence = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5D`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Requests executees :

| request | family | target | control | target_signal | control_signal | support_events | contradiction_events |
|---|---|---|---|---:|---:|---:|---:|
| `...::0001` | `object_delta_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0003` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0005` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0006` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::0007` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0008` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::0009` | `local_patch_change_candidate` | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::0010` | `local_patch_change_candidate` | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |

Note de mesure :

- Les requests `local_patch_before_after` sans coordonnees target explicites
  gardent la metrique demandee, mais l'execution annote un fallback de signal
  `changed_pixels_fallback_for_unparameterized_local_patch`.
- Ce fallback evite de transformer une action non parametree comme `ACTION5`
  en faux neutre technique.
- Le fallback est une mesure candidate-only, pas une preuve.

Lecture :

SAGE.5d valide la premiere boucle scientifique unknown-game end-to-end :
SAGE peut partir d'un etat live inconnu, produire une request M3 via SAGE.5c,
rejouer exactement le prefixe, verifier le hash de contexte, executer target et
controle dynamique, puis mesurer un effet local. Les 8 support_events indiquent
que les effets target sont superieurs aux controles dans ces contextes, mais ils
ne sont pas convertis en support scientifique.

Limites :

- Seulement 8/20 requests SAGE.5c sont executees.
- Les 20 requests SAGE.5c viennent encore du budget 50, car le cap SAGE.5c est
  epuise avant les budgets 150/300.
- `levels_completed` reste absent de la validation.
- La competence de jeu n'est pas validee.
- A32/A33 restent inchanges.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.live_mini_frontier_m3_executor `
  --source-sage5c diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5d_live_mini_frontier_m3_results.json `
  --min-requests 4 `
  --max-requests 8
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5e a maintenant ete execute. Il corrige la limite principale de SAGE.5c :
la generation n'est plus concentree sur le budget 50. Le signal candidate-only
est distribue sur plusieurs profondeurs de trajectoire, avec dedup strict et
execution M3 live-prefix sur chaque budget.

## SAGE.5e - Distributed live mini-frontier generation

Objectif :

- Lire `SAGE.5`, `SAGE.5b` et `SAGE.5c` comme contexte candidate-only.
- Generer une mini-frontier live inconnue avec quota par budget.
- Deduplicater par `(context_hash, target_action, target_args, diff_signature)`.
- Prioriser les contextes non redondants.
- Executer un petit nombre de requests par budget en `LIVE_PREFIX_REPLAY_CONTEXT`.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/distributed_live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_distributed_live_mini_frontier_generation.py`
- `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`

Run du 2026-07-18 :

- `game_id = sb26-7fbdac44`
- budgets = `[50, 150, 300]`
- `requests_per_budget = 8`
- `execute_requests_per_budget = 2`
- `distributed_generation = true`
- `dedup_policy = context_hash,target_action,target_args,diff_signature`
- `rerun_m2_m3_requested = 297`
- `effective_requests_generated = 24`
- `effective_request_ratio = 0.080808`
- `source_sage5c_effective_requests_generated = 20`
- `mini_frontier_hypotheses_generated = 24`
- `mini_frontier_m3_requests = 24`
- `dedup_key_count = 24`
- `duplicate_candidates_skipped = 24`
- `selected_execution_requests = 6`
- `requests_executed = 6`
- `requests_blocked = 0`
- `live_prefix_replay_exact_events = 6`
- `context_snapshot_hash_verified_events = 6`
- `support_events = 6`
- `contradiction_events = 0`
- `neutral_events = 0`
- `blocked_replay_events = 0`
- `execution_failures = 0`
- `outcome_status = SAGE_DISTRIBUTED_LIVE_MINI_FRONTIER_GENERATED_AND_EXECUTED_CANDIDATE_ONLY`
- `gate_passed = true`
- `generated_requests_counted_as_support = false`
- `mini_frontier_counted_as_evidence = false`
- `mini_frontier_execution_counted_as_evidence = false`
- `support_events_counted_as_support = false`
- `policy_result_counted_as_confirmation = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5E`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Par budget :

| budget | placeholders | generated | effective_ratio | duplicates_skipped | execution_selected | executed |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 47 | 8 | 0.170213 | 0 | 2 | 2 |
| 150 | 125 | 8 | 0.064 | 8 | 2 | 2 |
| 300 | 125 | 8 | 0.064 | 16 | 2 | 2 |

Familles :

- `families_generated = {"local_patch_change_candidate":23, "object_delta_candidate":1}`
- `families_executed = {"local_patch_change_candidate":5, "object_delta_candidate":1}`
- `target_actions_generated = {"ACTION5":13, "ACTION6":11}`
- `target_actions_executed = {"ACTION5":3, "ACTION6":3}`

Requests executees :

| request | family | budget | target | control | target_signal | control_signal | support_events | contradiction_events |
|---|---|---:|---|---|---:|---:|---:|---:|
| `...::050::0001` | `object_delta_candidate` | 50 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::050::0006` | `local_patch_change_candidate` | 50 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::150::0011` | `local_patch_change_candidate` | 150 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::150::0012` | `local_patch_change_candidate` | 150 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |
| `...::300::0019` | `local_patch_change_candidate` | 300 | `ACTION5` | `ACTION6` | 21 | 4 | 1 | 0 |
| `...::300::0020` | `local_patch_change_candidate` | 300 | `ACTION6` | `ACTION5` | 5 | 1 | 1 | 0 |

Lecture :

SAGE.5e valide la repartition de la mini-frontier live sur plusieurs profondeurs
de trajectoire inconnue. La chaine reste strictement candidate-only : les
support_events locaux montrent seulement que les mesures target depassent les
controles dynamiques dans ces contextes replayes, mais ils ne sont pas convertis
en support scientifique.

La correction principale par rapport a SAGE.5c est nette : SAGE.5c generait
20/20 requests sur le budget 50, alors que SAGE.5e genere 8 requests sur chacun
des budgets 50, 150 et 300, puis execute 2 requests par budget avec replay exact
et hash de contexte verifie.

Limites :

- Le test reste sur un seul jeu inconnu : `sb26-7fbdac44`.
- `levels_completed` reste hors validation ; aucune competence de jeu n'est
  confirmee.
- Les familles restent dominees par `local_patch_change_candidate` avec une
  seule `object_delta_candidate`.
- Les 6 support_events sont locaux, potentiellement redondants, et ne comptent
  ni comme evidence ni comme support.
- Des placeholders restent non transformes : la boucle apprenante existe, mais
  elle n'est pas encore continue.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.distributed_live_mini_frontier_generation `
  --m2-fused-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-fused-results diagnostics\m3\fused_llm_wm_experiment_results.json `
  --m3-counterfactual-feasibility diagnostics\m3\offline_frame_counterfactual_feasibility.json `
  --p1-policy-probe diagnostics\p1\bp35_sage_candidate_policy_probe.json `
  --p1-utility-handoff diagnostics\p1\bp35_candidate_policy_utility_handoff.json `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-sage5b diagnostics\sage\sage5b_switch_attribution_placeholder_audit.json `
  --source-sage5c diagnostics\sage\sage5c_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --requests-per-budget 8 `
  --execute-requests-per-budget 2 `
  --max-counterfactual-collections 8 `
  --progress-stall-window 8 `
  --same-action-arg-repeats 4 `
  --low-state-novelty-threshold 3 `
  --repeated-action-arg-rate-threshold 0.75
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_sage_long_horizon_progress_stall_transfer.py `
  tests\test_sage_progress_stall_trigger.py `
  tests\test_sage_long_horizon_transfer.py `
  tests\test_sage_subgoal_switcher.py `
  tests\test_sage_known_game_policy_probe.py `
  tests\test_sage_policy_loop_guard.py `
  tests\test_sage_live_prefix_counterfactual_collector.py `
  tests\test_sage_known_game_closed_loop_runner.py `
  tests\test_sage_known_game_closed_loop_scaffold.py -q
```

Transition :

SAGE.5f a maintenant ete execute. Il ne genere pas de nouvelles observations et
ne rejoue pas l'environnement : il consolide les observations SAGE.5e en motifs
candidate-only pour distinguer support local, redondance, robustesse multi-budget
et frontieres candidates A32-review sans write.

## SAGE.5f - Mini-frontier event consolidation

Objectif :

- Lire `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`.
- Verifier que la source SAGE.5e reste candidate-only.
- Construire des event records normalises depuis les executions M3 live-prefix.
- Regrouper les observations par regularite :
  - famille d'hypothese
  - action cible et arguments
  - transitions de couleurs
  - taille de patch / `changed_cells`
  - deltas de composantes
  - terminal / level delta
- Mesurer la stabilite par budget, action et contexte.
- Emettre seulement des clusters candidate-only, jamais un verdict.
- Garder `support=0`, aucun write A32/A33.

Ajouts :

- `theory/sage/mini_frontier_event_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_mini_frontier_event_consolidation.py`
- `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json`

Run du 2026-07-18 :

- source = `diagnostics/sage/sage5e_distributed_live_mini_frontier_results.json`
- `source_requests_generated = 24`
- `source_requests_executed = 6`
- `event_records = 6`
- `candidate_mechanism_clusters = 3`
- `multi_budget_clusters = 2`
- `robust_multi_budget_clusters = 2`
- `ready_for_A32_review_candidates = 2`
- `unique_contexts = 6`
- `unique_dedup_keys = 6`
- `raw_support_events = 6`
- `raw_contradiction_events = 0`
- `raw_neutral_events = 0`
- `actions_covered = {"ACTION5":3, "ACTION6":3}`
- `families_covered = {"local_patch_change_candidate":5, "object_delta_candidate":1}`
- `candidate_status_counts = {"LOCAL_SUPPORT_CANDIDATE_ONLY":1, "ROBUST_MULTI_BUDGET_CANDIDATE_ONLY":2}`
- `outcome_status = SAGE_MINI_FRONTIER_EVENTS_CLUSTERED_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5F`
- `revision_status = CANDIDATE_ONLY`
- `clustered_support_events_counted_as_support = false`
- `cluster_status_counted_as_scientific_verdict = false`
- `candidate_a32_frontier_counted_as_revision = false`
- `ready_for_A32_review_is_not_verdict = true`
- `a32_write_performed = false`
- `a33_write_performed = false`

Clusters :

| cluster | family | action | budgets | status | raw_support | contradictions | pattern |
|---|---|---|---|---|---:|---:|---|
| `001` | `local_patch_change_candidate` | `ACTION6 {"x":26,"y":57}` | 50/150/300 | `ROBUST_MULTI_BUDGET_CANDIDATE_ONLY` | 3 | 0 | `4->0` x20, `changed_cells=20`, effect size 4 |
| `002` | `object_delta_candidate` | `ACTION5` | 50 | `LOCAL_SUPPORT_CANDIDATE_ONLY` | 1 | 0 | `0->4` x20 + `2->3` x1, effect size 17 |
| `003` | `local_patch_change_candidate` | `ACTION5` | 150/300 | `ROBUST_MULTI_BUDGET_CANDIDATE_ONLY` | 2 | 0 | `0->4` x20 + `2->3` x1, `changed_cells=21`, effect size 17 |

Frontieres candidates A32-review :

- `sage5f::candidate_a32_frontier::001` derive du cluster `001`.
- `sage5f::candidate_a32_frontier::002` derive du cluster `003`.
- Ces frontieres sont marquees `ready_for_A32_review=true`, mais
  `ready_for_A32_review_is_not_verdict=true` et
  `candidate_a32_frontier_counted_as_revision=false`.
- Aucun fichier A32/A33 n'est modifie.

Lecture :

SAGE.5f valide la premiere memoire experimentale candidate-only sur jeu inconnu :
les observations locales SAGE.5e ne restent plus une liste plate de support_events.
Elles sont regroupees en motifs d'effet, avec une distinction entre support local
single-budget et robustesse multi-budget.

Deux motifs sont assez stables pour etre proposes a une revue A32 candidate-only :
`ACTION6` produit un motif local `4->0` sur les budgets 50/150/300, et `ACTION5`
produit un motif `0->4` + `2->3` sur les budgets 150/300. Cela ne confirme pas
une mecanique : c'est seulement une consolidation structurante pour une future
revue.

Limites :

- Le test reste sur `sb26-7fbdac44` uniquement.
- Les clusters viennent de 6 executions seulement.
- Les 2 frontieres A32-review candidates ne sont pas des decisions A32.
- `levels_completed` reste absent : aucune competence de jeu n'est validee.
- Les effets sont locaux et peuvent encore etre redondants ou contextuels.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.mini_frontier_event_consolidation `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5f_mini_frontier_event_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5g a maintenant ete compile. Il transforme les deux frontieres robustes de
SAGE.5f en dossiers auditables pour A32, sans executer de nouveau test, sans
produire de verdict et sans ecrire dans A32/A33. Les lacunes historiques A32
sont explicites et converties en demandes de suivi controlees.

## SAGE.5g - A32 review handoff compiler

Objectif :

- Lire `diagnostics/sage/sage5f_mini_frontier_event_consolidation.json`.
- Verifier que la source SAGE.5f reste candidate-only et `support=0`.
- Compiler chaque cluster robuste en dossier A32-review auditable.
- Conserver les contextes, budgets, interventions cibles, controles et motifs
  d'effet observes.
- Comparer chaque dossier aux seuils historiques A32 :
  - au moins 3 support events bruts ;
  - au moins 2 contextes replay-exacts ;
  - zero contradiction ;
  - au moins 2 actions de controle distinctes.
- Lier les clusters proches sans les fusionner automatiquement.
- Produire des followups precis, sans execution ni revision.
- Garder `support=0`, aucun verdict, aucun write A32/A33.

Ajouts :

- `theory/sage/a32_review_handoff.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_a32_review_handoff.py`
- `diagnostics/sage/sage5g_a32_review_handoff.json`

Run du 2026-07-18 :

- `source_candidate_mechanism_clusters = 3`
- `source_ready_for_A32_review_candidates = 2`
- `handoff_items = 2`
- `items_ready_for_A32_review = 2`
- `items_without_followup_requirements = 0`
- `followup_requests = 4`
- `raw_support_events_in_handoff = 5`
- `raw_contradiction_events_in_handoff = 0`
- `independent_context_events_in_handoff = 5`
- `related_nonmerged_cluster_links = 1`
- `outcome_status = SAGE_A32_REVIEW_HANDOFF_COMPILED_CANDIDATE_ONLY`
- `gate_passed = true`
- `candidate_review_item_counted_as_revision = false`
- `independent_context_events_counted_as_scientific_support = false`
- `execution_performed = false`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5G`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Dossiers :

| candidat | support brut | contextes | controles distincts | recommandation |
|---|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | 3 | 3 | 1 (`ACTION5`) | `FOLLOWUP_REQUIRED_CONTROL_DIVERSITY` |
| `ACTION5` | 2 | 2 | 1 (`ACTION6`) | `FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY` |

Followups :

- `ACTION6` : retester contre une action de controle autre que `ACTION5`, dans
  au moins deux contextes replay-exacts.
- `ACTION5` : obtenir une troisieme observation comparable.
- `ACTION5` : retester contre une action de controle autre que `ACTION6`.
- `ACTION5` : remesurer les clusters `002` et `003` avec les lectures
  `local_patch` et `object_delta`, en les gardant non fusionnes avant revue.

Lecture :

SAGE.5g ne confirme ni `ACTION6` ni `ACTION5`. Il ferme le contrat de handoff :
un motif robuste devient un dossier falsifiable, ses insuffisances deviennent
des demandes d'experience explicites, et A32 reste le premier composant autorise
a produire une decision scientifique.

Les cinq contextes cumules ne sont pas transformes en support scientifique. Ils
servent uniquement a documenter la diversite des observations. De meme, les
budgets 50/150/300 ne sont jamais comptes comme des actions de controle
distinctes.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.a32_review_handoff `
  --source-sage5f diagnostics\sage\sage5f_mini_frontier_event_consolidation.json `
  --out diagnostics\sage\sage5g_a32_review_handoff.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5h a maintenant execute les quatre followups SAGE.5g. Deux acquisitions
sont obtenues et deux sont bloquees avec une raison structurelle reproductible :
aux contextes replayes, les seules familles d'action legales sont `ACTION5` et
`ACTION6`. Apres exclusion de l'action cible et du controle deja utilise, aucune
troisieme action distincte n'est disponible.

## SAGE.5h - Controlled follow-up acquisition

Objectif :

- Lire les dossiers et followups SAGE.5g.
- Retrouver les requests replayables SAGE.5e et les clusters SAGE.5f.
- Auditer la surface d'actions legales avant toute acquisition de nouveau
  controle.
- Exiger un replay exact et un hash de contexte identique.
- Acquerir un troisieme evenement comparable pour `ACTION5`.
- Remesurer les clusters lies `002` et `003` avec :
  - `local_patch_before_after` ;
  - `object_counts_before_after` comme lecture object-delta.
- Garder les clusters non fusionnes en cas de divergence de mesure.
- Resoudre chaque followup comme acquis ou bloque avec une raison auditable.
- Garder `support=0`, aucune revision, aucun write A32/A33.

Ajouts :

- `theory/sage/controlled_followup_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_controlled_followup_acquisition.py`
- `diagnostics/sage/sage5h_controlled_followup_acquisition.json`

Run du 2026-07-18 :

- `followup_requests_consumed = 4`
- `followup_outcomes = 4`
- `followups_completed = 2`
- `followups_blocked = 2`
- `control_diversity_followups_completed = 0`
- `control_diversity_followups_blocked = 2`
- `support_followups_completed = 1`
- `cross_measurement_followups_completed = 1`
- `control_surface_contexts_audited = 3`
- `replay_exact_control_surface_audits = 3`
- `control_surface_exhausted_audits = 3`
- `controlled_experiments_executed = 4`
- `controlled_experiments_blocked = 0`
- `live_prefix_replay_exact_experiments = 4`
- `raw_support_events = 2`
- `raw_contradiction_events = 0`
- `raw_neutral_events = 2`
- `comparable_support_events_acquired = 1`
- `cross_measurement_alignments = 0`
- `cross_measurement_divergences = 1`
- `candidates_ready_for_A32_intake = 0`
- `all_followups_resolved = true`
- `all_requested_followups_completed = false`
- `outcome_status = SAGE_CONTROLLED_FOLLOWUP_ACQUISITION_PARTIAL_CONTROL_SURFACE_LIMIT_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5H`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Resolution des followups :

| candidat | followup | resolution |
|---|---|---|
| `ACTION6 {"x":26,"y":57}` | controle distinct dans 2 contextes | `BLOCKED_NO_DISTINCT_LEGAL_CONTROL_ACTION` |
| `ACTION5` | troisieme support comparable | `ACQUIRED_COMPARABLE_SUPPORT_CANDIDATE_ONLY` |
| `ACTION5` | controle distinct | `BLOCKED_NO_DISTINCT_LEGAL_CONTROL_ACTION` |
| `ACTION5` | remesure clusters 002/003 | `ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE_CANDIDATE_ONLY` |

Surface de controle :

- Pour `ACTION6`, les contextes audites exposent uniquement `ACTION5` et
  plusieurs variantes parametrees de `ACTION6`.
- Pour `ACTION5`, le contexte audite expose uniquement `ACTION5` et plusieurs
  variantes parametrees de `ACTION6`.
- `RESET` reste exclu des controles scientifiques.
- Les variantes d'arguments d'`ACTION6` ne sont pas comptees comme de nouvelles
  actions de controle distinctes par le contrat A32 historique.

Remesure croisee `ACTION5` :

- Lecture locale alignee sur les deux clusters :
  - `changed_pixels = 21` ;
  - `local_patch_available = false` ;
  - `local_changed_pixels = 0`.
- Lecture object-delta divergente :
  - cluster `002` : `object_count_delta = 0`, deltas couleur
    `{"0":-1,"3":1}` ;
  - cluster `003` : `object_count_delta = -1`, delta couleur `{"0":-1}`.
- Les clusters restent donc explicitement lies mais non fusionnes.
- Cette divergence n'est ni une contradiction scientifique ni une refutation :
  elle montre que l'effet object-centric depend du contexte.

Mise a jour des dossiers :

| candidat | support brut avant | nouveau support comparable | support brut apres | contextes | controles distincts | statut |
|---|---:|---:|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | 3 | 0 | 3 | 3 | 1 | controle distinct indisponible |
| `ACTION5` | 2 | 1 | 3 | 3 | 1 | support atteint, controle distinct indisponible |

Les deux dossiers restent `CANDIDATE_ONLY` et ne sont pas prets pour une intake
A32 confirmatoire avec les seuils historiques.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.controlled_followup_acquisition `
  --source-sage5g diagnostics\sage\sage5g_a32_review_handoff.json `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --source-sage5f diagnostics\sage\sage5f_mini_frontier_event_consolidation.json `
  --out diagnostics\sage\sage5h_controlled_followup_acquisition.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_controlled_followup_acquisition.py `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Transition :

SAGE.5i a maintenant parcouru toute la frontiere candidate SAGE.5e rejouable,
et non plus seulement les trois contextes necessaires aux followups SAGE.5h.
Les 24 contextes ont ete reproduits exactement. Aucun ne rend legale une
troisieme famille d'action : la surface action-distincte est donc epuisee dans
ce perimetre borne, sans que cet epuisement constitue une refutation globale.

## SAGE.5i - Control-surface expansion

Objectif :

- Lire les artefacts candidate-only SAGE.5e, SAGE.5g et SAGE.5h.
- Selectionner toutes les requests SAGE.5e qui correspondent exactement aux
  deux signatures candidates SAGE.5g.
- Rejouer chaque contexte et exiger un hash identique avant d'auditer les
  actions legales.
- Executer un nouveau controle dans au moins deux contextes si une troisieme
  famille d'action est disponible.
- Sinon, produire une preuve d'epuisement limitee a cette frontiere rejouable.
- Inventorier separement les variantes parametrees sans les compter comme des
  familles d'action distinctes.
- Proposer les choix protocolaires a A32 sans modifier A32/A33 ni produire de
  verdict automatique.
- Garder `support=0`, aucune revision et aucun write A32/A33.

Ajouts :

- `theory/sage/control_surface_expansion.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_control_surface_expansion.py`
- `diagnostics/sage/sage5i_control_surface_expansion.json`

Run du 2026-07-18 :

- `source_requests_available = 24`
- `candidates_evaluated = 2`
- `candidate_contexts_considered = 24`
- `unique_candidate_contexts = 24`
- `context_surface_audits = 24`
- `replay_exact_context_surface_audits = 24`
- `action_surface_signature_counts = {"ACTION5,ACTION6":24}`
- `candidate_context_counts = {"ACTION5":13,"ACTION6":11}`
- `contexts_with_third_action_family = 0`
- `candidates_with_action_distinct_surface_exhausted = 2`
- `candidates_with_new_distinct_control = 0`
- `control_expansion_experiments_executed = 0`
- `parameterized_control_option_counts = {"ACTION5":4,"ACTION6":3}`
- `candidates_ready_for_A32_intake = 0`
- `bounded_action_distinct_exhaustion_proven = true`
- `bounded_scope_only = true`
- `all_candidate_contexts_audited_exact = true`
- `a32_parameterized_protocol_proposal_generated = true`
- `outcome_status = SAGE_CONTROL_SURFACE_EXPANSION_ACTION_DISTINCT_EXHAUSTED_CANDIDATE_ONLY`
- `gate_passed = true`
- `support = 0`
- `truth_status = NOT_EVALUATED_BY_SAGE_5I`
- `revision_status = CANDIDATE_ONLY`
- `a32_write_performed = false`
- `a33_write_performed = false`

Surface auditee :

| candidat | contextes exacts | familles legales | troisieme famille | options parametrees |
|---|---:|---|---:|---:|
| `ACTION6 {"x":26,"y":57}` | 11 | `ACTION5`, `ACTION6` | 0 | 3 |
| `ACTION5` | 13 | `ACTION5`, `ACTION6` | 0 | 4 |

Les trois options du candidat `ACTION6` sont les autres positions
`ACTION6 {"x":18|34|42,"y":57}`. Les quatre options du candidat `ACTION5`
sont `ACTION6 {"x":21|27|33|39,"y":28}`. Elles sont conservees comme
interventions parametrees potentielles, jamais comme nouvelles familles
d'action.

Conclusion bornee :

- SAGE.5i prouve uniquement qu'aucune troisieme famille d'action n'est
  disponible dans les 24 contextes candidats rejouables de SAGE.5e.
- Cette preuve ne couvre ni tous les etats atteignables de `sb26`, ni un autre
  jeu, et ne compte pas comme refutation.
- Les deux candidats conservent 3 observations brutes, 3 contextes independants,
  zero contradiction et une seule famille de controle distincte.
- Le seuil historique A32 de deux familles de controle reste donc non satisfait.
- Aucune variante parametree n'est requalifiee automatiquement et aucune intake
  confirmatoire A32 n'est declenchee.

Proposition protocolaire A32 :

- Conserver le seuil action-distinct et garder les candidats non resolus ; ou
- autoriser explicitement un protocole pre-enregistre de controles parametres ;
  ou
- rejeter les candidats comme non identifiables dans la surface actuelle.
- En cas d'autorisation parametree : arguments distincts de la cible, replay
  exact dans des contextes distincts, au moins deux variantes et zero
  contradiction, sans les renommer en familles d'action distinctes.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.control_surface_expansion `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --source-sage5g diagnostics\sage\sage5g_a32_review_handoff.json `
  --source-sage5h diagnostics\sage\sage5h_controlled_followup_acquisition.json `
  --out diagnostics\sage\sage5i_control_surface_expansion.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_control_surface_expansion.py `
  tests\test_sage_controlled_followup_acquisition.py `
  tests\test_sage_a32_review_handoff.py `
  tests\test_sage_mini_frontier_event_consolidation.py `
  tests\test_sage_distributed_live_mini_frontier_generation.py `
  tests\test_sage_live_mini_frontier_m3_executor.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py `
  tests\test_sage_unknown_game_bounded_probe.py -q
```

Decision A32.4 du 2026-07-18 :

- A32.4 choisit
  `AUTHORIZE_PRE_REGISTERED_PARAMETERIZED_CONTROL_PROTOCOL` pour les deux
  candidats.
- L'exigence action-distincte reste la regle par defaut ; l'autorisation est une
  exception candidate-specifique et bornee.
- Deux variantes extremes sont pre-enregistrees par candidat :
  - cible `ACTION6 {"x":26,"y":57}` : controles `x=18` et `x=42` ;
  - cible `ACTION5` : controles `ACTION6 x=21` et `ACTION6 x=39`, `y=28`.
- Chaque variante doit etre executee dans deux contextes replay-exacts, budgets
  50 et 300 : 8 experiences demandees au total.
- Aucune experience n'est executee par A32.4 ; les deux hypotheses restent
  unresolved, `support=0`, non A33-ready.
- Artefact :
  `diagnostics/a32/unknown_game_control_protocol_decisions.json`.

Transition :

SAGE.5j a execute les huit experiences A32.4 exactement comme pre-enregistrees.
Aucune variante, aucun contexte, aucun argument ni aucune mesure n'a ete
substitue. Les deux candidats donnent des resultats opposes mais parfaitement
stables sur leurs quatre repetitions : `ACTION6` est non discriminant face aux
autres positions `ACTION6`, tandis qu'`ACTION5` reste discriminant face aux
positions `ACTION6` selectionnees.

## SAGE.5j - Pre-registered parameterized control acquisition

Objectif :

- Lire la decision A32.4 et les requests replayables SAGE.5e.
- Verifier que les deux hypotheses A32.4 sont toujours unresolved, `support=0`,
  sans confirmation, refutation ni write A33.
- Consommer uniquement les huit experiences `PRE_REGISTERED_NOT_EXECUTED`.
- Faire correspondre chaque experience a sa request SAGE.5e par identifiant,
  cible, arguments, contexte, budget et mesure.
- Rejouer independamment les bras cible et controle depuis le meme contexte.
- Exiger les arguments exacts des controles `18/42` et `21/39`.
- Interdire toute substitution post-hoc de variante, contexte ou mesure.
- Classer chaque paire comme discriminante, non discriminante ou controle
  superieur, sans transformer ce resultat en verdict scientifique.
- Garder `support=0`, aucune revision, aucun write A32/A33.

Ajouts :

- `theory/sage/parameterized_control_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_parameterized_control_acquisition.py`
- `diagnostics/sage/sage5j_parameterized_control_acquisition.json`

Run du 2026-07-18 :

- `source_protocol_decisions=2`
- `pre_registered_experiments_consumed=8`
- `experiments_executed=8`
- `experiments_blocked=0`
- `live_prefix_replay_exact_experiments=8`
- `protocol_exact_match_experiments=8`
- `protocol_substitutions_detected=0`
- `target_control_pairs_executed=8`
- `parameterized_variants_executed=4`
- `variant_replications_completed=4`
- `raw_support_events=4`
- `raw_contradiction_events=0`
- `raw_neutral_events=4`
- `candidates_evaluated=2`
- `candidate_protocols_complete=2`
- `candidates_with_discriminating_parameterized_controls=1`
- `candidates_with_non_discriminating_parameterized_controls=1`
- `candidates_with_control_exceeding_target=0`
- `candidates_ready_for_A32_protocol_result_review=2`
- `all_pre_registered_experiments_executed_exactly=true`
- `gate_passed=true`
- `outcome_status=SAGE_PARAMETERIZED_CONTROL_ACQUISITION_MIXED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_5J`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats par candidat :

| candidat | controles parametres | paires exactes | signal cible | signal controle | resultat |
|---|---|---:|---:|---:|---|
| `ACTION6 {"x":26,"y":57}` | `ACTION6 x=18/42,y=57` | 4 | 5 | 5 | `PARAMETERIZED_CONTROLS_NON_DISCRIMINATING_CANDIDATE_ONLY` |
| `ACTION5` | `ACTION6 x=21/39,y=28` | 4 | 21 | 4 | `PARAMETERIZED_CONTROLS_DISCRIMINATING_CANDIDATE_ONLY` |

Lecture scientifique :

- Les quatre controles parametres du candidat `ACTION6` reproduisent exactement
  le meme signal local que la cible. Le protocole n'identifie donc pas un effet
  specifique a `x=26`.
- Les quatre controles du candidat `ACTION5` restent nettement differents de la
  cible, avec un delta apparie constant de `17`.
- Les evenements bruts `4 support / 4 neutral / 0 contradiction` ne sont pas
  comptes comme support scientifique par SAGE.
- Les evenements neutres `ACTION6` ne sont pas comptes comme refutation.
- Les variantes parametrees ne sont toujours pas renommees en familles d'action
  distinctes.
- Les deux dossiers sont complets et prets pour une revue A32 du resultat du
  protocole ; A32 reste seul autorise a decider.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.parameterized_control_acquisition `
  --source-a32-4 diagnostics\a32\unknown_game_control_protocol_decisions.json `
  --source-sage5e diagnostics\sage\sage5e_distributed_live_mini_frontier_results.json `
  --out diagnostics\sage\sage5j_parameterized_control_acquisition.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_parameterized_control_acquisition.py `
  tests\test_a32_unknown_game_control_protocol_decisions.py `
  tests\test_sage_control_surface_expansion.py `
  tests\test_sage_controlled_followup_acquisition.py -q
```

Retour A32.5 du 2026-07-18 :

- A32.5 confirme ACTION5 uniquement dans le scope borne de `sb26-7fbdac44` :
  quatre paires exactes discriminantes, support scientifique `4`, aucune
  contradiction.
- A32.5 conserve `ACTION6 {"x":26,"y":57}` unresolved et non identifiable :
  les quatre controles ACTION6 parametres reproduisent le signal cible `5`.
- Cette non-discrimination n'est pas une refutation d'un eventuel effet ACTION6
  position-invariant.
- Un seul dossier, ACTION5, est pret pour revue du registre A33 ; A32.5
  n'effectue aucun write A33.
- Artefact :
  `diagnostics/a32/unknown_game_parameterized_control_revision_decisions.json`.

Retour A33.2 du 2026-07-18 :

- ACTION5 est enregistree dans
  `diagnostics/a33/scoped_unknown_game_registry.json` avec son jeu, son
  candidat, ses quatre contextes, sa mesure, ses budgets et ses deux controles
  verrouilles.
- Le support `4` provient exclusivement du verdict A32.5 ; A33.2 ne reconfirme
  pas la mecanique et ne generalise pas son scope.
- ACTION6 unresolved est exclue explicitement du registre confirme.
- Le registre A33.1 et ses consommateurs A34-A39 ne sont pas modifies.

## SAGE.6 - Second unknown-game bounded transfer

Objectif :

- Fermer explicitement la boucle source SAGE.5j -> A32.5 -> A33.2 avant de
  changer de jeu.
- Auditer les jeux `public_unseen` dans leur ordre fixe, sans lire de metrique
  d'outcome pour choisir le second jeu.
- Exiger pour le jeu selectionne : aucune trace humaine, aucune trace M2, aucun
  prior specifique et une identite differente de `sb26`.
- Rejouer la discipline bornee SAGE.5 sur budgets `50`, `150`, `300`.
- Verifier la legalite des actions, les switches, le trigger progress-stall, la
  repetition, le taux terminal et la nouveaute d'etat.
- Lire A33.2 uniquement comme garde de quarantaine : ne reutiliser ni ACTION5
  confirmee sur `sb26`, ni le candidat ACTION6 unresolved.
- Garder le resultat candidate-only, `support=0`, sans write A32/A33.

Selection pre-execution :

| rang fixe | jeu | hygiene unknown | decision |
|---:|---|---|---|
| 1 | `wa30-ee6fef47` | oui | selectionne |
| 2 | `tn36-ab4f63cc` | oui | eligible, non selectionne |
| 3 | `ft09-0d8bbf25` | non, traces humaines et M2 | exclu connu |
| 4 | `cn04-65d47d14` | non, traces humaines et M2 | exclu connu |
| 5 | `sb26-7fbdac44` | oui | exclu car jeu source |

La selection de `wa30` provient uniquement de
`PUBLIC_UNSEEN_FIXED_ORDER_BEFORE_EXECUTION`. Les resultats de `wa30` et
`tn36` n'interviennent jamais dans ce choix.

Ajouts :

- `theory/sage/second_unknown_game_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_transfer.py`
- `diagnostics/sage/sage6_second_unknown_game_transfer_results.json`

Run du 2026-07-18 :

- `source_game_id=sb26-7fbdac44`
- `selected_second_game_id=wa30-ee6fef47`
- `candidate_games_audited=5`
- `eligible_unknown_games=2`
- `known_or_seen_candidates_excluded=2`
- `source_or_registry_scope_candidates_excluded=1`
- `budgets_evaluated=[50,150,300]`
- `budgets_gate_passed=3`
- `budgets_total=3`
- `all_budgets_gate_passed=true`
- `budgets_with_progress_stall_detected=[50,150,300]`
- `subgoal_switches_total=98`
- `max_repeated_action_arg_rate=0.5`
- `max_terminal_rate=0.005`
- `min_unique_state_signatures=37`
- `max_unique_state_signatures=132`
- `levels_completed_max=0`
- `source_a33_2_entries_quarantined=1`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`
- `wrong_confirmations=0`

Resultats par budget :

| budget | gate | switches | progress-stall | repetition | terminal | signatures uniques | niveaux |
|---:|---|---:|---|---:|---:|---:|---:|
| 50 | passe | 12 | oui, 12 switches | 0.48 | 0.0 | 37 | 0 |
| 150 | passe | 37 | oui, 37 switches | 0.493333 | 0.0 | 101 | 0 |
| 300 | passe | 49 | oui, 49 switches | 0.5 | 0.005 | 132 | 0 |

Lecture :

- SAGE.6 etend le probe technique borne a un deuxieme jeu inconnu reel. Il ne
  constitue toujours ni un benchmark ni une preuve de competence de jeu.
- Contrairement a `sb26`, le trigger progress-stall est effectivement detecte
  sur les trois budgets de `wa30`. SAGE.6 fournit donc une seconde validation
  de transfert de la discipline de boucle, sur une famille warehouse/navigation.
- La repetition reste sous le seuil catastrophique `0.9` et le taux terminal
  sous `0.05`, mais `levels_completed=0` : aucune reussite ludique n'est
  revendiquee.
- L'entree ACTION5 A33.2 reste verrouillee a `sb26`; aucune action, mecanique ou
  evidence de cette memoire n'est appliquee a `wa30`.
- Les observations `wa30` restent candidate-only et ne sont pas du support
  scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_transfer `
  --source-sage5 diagnostics\sage\sage5_unknown_game_bounded_probe_results.json `
  --source-a33-2 diagnostics\a33\scoped_unknown_game_registry.json `
  --m2-dataset-manifest diagnostics\m2\arc_lewm_dataset_manifest.json `
  --human-traces-dir human_traces `
  --budgets 50 150 300 `
  --out diagnostics\sage\sage6_second_unknown_game_transfer_results.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_sage_unknown_game_bounded_probe.py `
  tests\test_a33_scoped_unknown_game_registry.py -q
```

## SAGE.6a - Second-game switch attribution and live mini-frontier

Objectif :

- Reproduire exactement les 98 switches `wa30` de SAGE.6 sur les budgets
  pre-enregistres `50`, `150`, `300`.
- Attribuer chaque switch a son trigger et a son sous-but, sans confondre une
  garde terminale avec le compteur de switches source.
- Mesurer la dependance au placeholder `rerun_m2_m3`.
- Rejouer les prefixes live correspondants et convertir un sous-ensemble borne
  de ces placeholders en hypotheses et requests M3 testables.
- Preserver la quarantaine A33.2 de `sb26`, `support=0` et l'absence de write
  A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_switch_frontier.py`
- generalisation parametree, retrocompatible SAGE.5c, de
  `theory/sage/live_mini_frontier_generation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_switch_frontier.py`
- `diagnostics/sage/sage6a_switch_attribution_mini_frontier.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets_evaluated=[50,150,300]`
- `source_switches_expected=98`
- `switches_reproduced=98`
- `source_switch_count_reproduced_exactly=true`
- `switches_due_to_progress_stall_or_repeat_collapse=98`
- `true_exploratory_switches=66`
- `active_counterfactual_switches=32`
- `reposition_switches=34`
- `placeholder_rerun_m2_m3_switches=32`
- `source_placeholder_switch_ratio=0.326531`
- `source_placeholder_dependency_under_threshold=true`
- `total_switch_events_observed=99`
- `terminal_guard_events_outside_source_switch_count=1`
- `max_generated_requests=20`
- `mini_frontier_hypotheses_generated=20`
- `effective_requests_generated=20`
- `effective_request_ratio=0.625`
- `unresolved_placeholder_switches_after_generation=12`
- `residual_placeholder_switch_ratio=0.122449`
- `unique_context_snapshot_hashes=12`
- `all_requests_ready_for_m3=true`
- `all_requests_live_prefix_replayable=true`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_SWITCH_FRONTIER_GENERATED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6A`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Attribution et generation par budget :

| budget | switches source | evenements observes | contrefactuels | repositionnements | placeholders | requests generees | placeholders residuels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 12 | 12 | 4 | 4 | 4 | 4 | 0 |
| 150 | 37 | 37 | 12 | 13 | 12 | 12 | 0 |
| 300 | 49 | 50 | 16 | 17 | 16 | 4 | 12 |

Le cinquantieme evenement du budget 300 est un `stop_safe_hold` terminal. Il
est conserve dans l'audit mais reste hors des 49 switches de sous-objectif
comptes par SAGE.6. Cette separation explique les 99 evenements observes sans
alterer la reproduction exacte des 98 switches source.

Les 20 hypotheses appartiennent a la famille candidate
`local_patch_change_candidate` et ciblent `ACTION2`. Leur distribution suit
l'ordre fixe des budgets sous un plafond global : 4 au budget 50, 12 au budget
150 et 4 au budget 300. Chaque request porte un prefixe live rejouable, un hash
de contexte et le statut `READY_FOR_M3`. Les 20 generations ne sont ni du
support, ni une evidence, ni une confirmation.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_switch_frontier `
  --source-sage6 diagnostics\sage\sage6_second_unknown_game_transfer_results.json `
  --budgets 50 150 300 `
  --max-generated-requests 20 `
  --out diagnostics\sage\sage6a_switch_attribution_mini_frontier.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_switch_frontier.py `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_sage_live_mini_frontier_generation.py `
  tests\test_sage_switch_attribution_placeholder_audit.py -q
```

## SAGE.6b - Stratified second-game M3 execution

Objectif :

- Selectionner exactement deux requests SAGE.6a par budget `50`, `150`, `300`.
- Maximiser la diversite globale des hashes de contexte, puis la dispersion des
  steps, sans consulter les resultats d'execution.
- Rejouer le meme prefixe live pour la cible et le controle dynamique.
- Exiger la verification du hash avant chaque action cible ou controle.
- Executer `ACTION2` contre le premier controle distinct pre-enregistre,
  `ACTION1`, et conserver les deltas comme evenements bruts candidate-only.
- Preserver la quarantaine A33.2, `support=0` et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_m3_execution.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_m3_execution.py`
- `diagnostics/sage/sage6b_second_unknown_game_m3_execution.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets_available=[50,150,300]`
- `requests_available=20`
- `requests_per_budget=2`
- `requests_selected=6`
- `requests_selected_by_budget={50:2,150:2,300:2}`
- `unique_selected_context_snapshot_hashes=6`
- `selected_source_steps_by_budget={50:[12,48],150:[132,144],300:[24,36]}`
- `requests_executed=6`
- `requests_executed_by_budget={50:2,150:2,300:2}`
- `requests_blocked=0`
- `live_prefix_replay_exact_events=6`
- `context_snapshot_hash_verified_events=6`
- `target_actions_executed={ACTION2:6}`
- `control_actions_executed={ACTION1:6}`
- `hypothesis_families_executed={local_patch_change_candidate:6}`
- `target_signal_total=194`
- `control_signal_total=34`
- `controlled_effect_sizes=[0,32,32,32,32,32]`
- `positive_effect_events=5`
- `zero_effect_events=1`
- `negative_effect_events=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `source_scoped_mechanics_reused=0`
- `cross_game_mechanics_imported=0`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_M3_EXECUTION_COMPLETED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6B`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats par budget :

| budget | steps | execution | replay/hash | signal cible | signal controle | effets bruts |
|---:|---|---:|---:|---:|---:|---|
| 50 | 12, 48 | 2/2 | 2/2 | 65 | 33 | 1 positif, 1 neutre |
| 150 | 132, 144 | 2/2 | 2/2 | 65 | 1 | 2 positifs |
| 300 | 24, 36 | 2/2 | 2/2 | 64 | 0 | 2 positifs |

La selection est pre-execution. Les budgets disposant du moins de contextes
sont alloues en premier pour que les budgets 50 et 300 se partagent leurs
quatre hashes communs. Le budget 150 fournit ensuite deux contexts tardifs
encore distincts. Les six hashes selectionnes sont donc uniques.

Les cinq `raw_support_events` indiquent seulement que, dans ces contextes live,
le signal local mesure apres ACTION2 depasse celui mesure apres ACTION1. Le
sixieme contexte donne un delta nul. SAGE.6b ne compte aucun de ces evenements
comme support scientifique, ne traite pas l'evenement neutre comme refutation
et ne produit aucun verdict.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_m3_execution `
  --source-sage6a diagnostics\sage\sage6a_switch_attribution_mini_frontier.json `
  --requests-per-budget 2 `
  --out diagnostics\sage\sage6b_second_unknown_game_m3_execution.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_m3_execution.py `
  tests\test_sage_second_unknown_game_switch_frontier.py `
  tests\test_sage_live_mini_frontier_m3_executor.py -q
```

## SAGE.6c - Context-preserving event consolidation

Objectif :

- Transformer les six executions SAGE.6b en enregistrements comparables sans
  rejouer l'environnement.
- Conserver exactement un cluster par hash de contexte live.
- Interdire toute fusion inter-contexte et utiliser les groupes d'effet comme
  index de comparaison uniquement.
- Evaluer la stabilite du delta ACTION2/ACTION1 entre budgets avec des seuils
  explicites de contextes, budgets et dispersion d'effet.
- Conserver l'evenement neutre comme exception contextuelle, jamais comme
  contradiction ou refutation.
- Bloquer la revue A32 tant que la diversite de controle reste insuffisante.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_event_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_event_consolidation.py`
- `diagnostics/sage/sage6c_second_unknown_game_event_consolidation.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `event_records=6`
- `context_clusters=6`
- `singleton_context_clusters=6`
- `cross_context_merges_performed=0`
- `all_contexts_preserved_without_merge=true`
- `effect_signature_groups=2`
- `stable_positive_multi_budget_groups=1`
- `neutral_effect_groups=1`
- `stable_positive_contexts=5`
- `stable_positive_events=5`
- `stable_positive_budgets=[50,150,300]`
- `stable_positive_effect_size=32`
- `stable_positive_effect_spread=0`
- `stable_positive_across_all_budgets=true`
- `neutral_context_exceptions=1`
- `negative_context_exceptions=0`
- `context_sensitive_exception_detected=true`
- `distinct_control_actions=1`
- `control_diversity_sufficient_for_a32_review=false`
- `candidate_handoff_frontiers=1`
- `ready_for_A32_handoff_compilation=1`
- `ready_for_A32_review=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_EVENTS_CONSOLIDATED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6C`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Clusters de contexte :

| cluster | budget | step | effet ACTION2-ACTION1 | statut candidate-only |
|---|---:|---:|---:|---|
| `001` | 50 | 12 | 0 | `NEUTRAL_CONTEXT_CANDIDATE_ONLY` |
| `002` | 50 | 48 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `003` | 150 | 132 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `004` | 150 | 144 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `005` | 300 | 24 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |
| `006` | 300 | 36 | +32 | `POSITIVE_CONTEXT_CANDIDATE_ONLY` |

Groupes de comparaison :

| groupe | contexts | budgets | effet | statut |
|---|---:|---|---:|---|
| `001` | 1 | 50 | 0 | `LOCAL_NEUTRAL_CANDIDATE_ONLY` |
| `002` | 5 | 50/150/300 | +32, dispersion 0 | `STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY` |

Le groupe `002` indexe cinq clusters distincts mais ne les fusionne pas. Le
cluster neutre `001` reste une exception explicite du meme protocole
ACTION2/ACTION1. Le statut global est donc
`STABLE_POSITIVE_MULTI_BUDGET_WITH_NEUTRAL_EXCEPTION_CANDIDATE_ONLY`.

La frontiere est prete pour compilation d'un handoff, pas pour revue A32. Un
seul controle distinct, ACTION1, a ete execute alors que le seuil est fixe a
deux. Les followups requis sont : ajouter un controle distinct par budget,
repliquer le contexte neutre et conserver les frontieres entre clusters.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_event_consolidation `
  --source-sage6b diagnostics\sage\sage6b_second_unknown_game_m3_execution.json `
  --min-stable-contexts 3 `
  --min-stable-budgets 2 `
  --max-positive-effect-spread 0 `
  --min-distinct-controls-for-a32-review 2 `
  --out diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py `
  tests\test_sage_mini_frontier_event_consolidation.py -q
```

La compilation candidate-only demandee ici est realisee par SAGE.6d ci-dessous.

SAGE.6c autorise maintenant a dire : SAGE sait distinguer un motif controle
stable multi-budget d'une exception contextuelle tout en preservant les
frontieres entre contexts. Il ne faut pas dire : SAGE a prouve une mecanique
ACTION2, que l'effet est universel sur `wa30`, ou qu'un verdict A32 est pret.

## SAGE.6d - Second-game handoff compiler

Objectif :

- Compiler la frontiere SAGE.6c en un dossier candidate-only executable lors
  d'une iteration ulterieure, sans rejouer l'environnement.
- Choisir de maniere deterministe le premier controle suggere qui soit distinct
  de la cible ACTION2 et du controle deja execute ACTION1.
- Pre-enregistrer une experience ACTION2/ACTION3 sur un contexte positif stable
  de chacun des budgets 50, 150 et 300.
- Pre-enregistrer une replication exacte ACTION2/ACTION1 du contexte neutre du
  budget 50.
- Embarquer pour chaque protocole le replay live complet, ses arguments, son
  hash de contexte et ses conditions d'interpretation fixees avant execution.
- Conserver les six clusters SAGE.6c dans un manifeste non fusionne.
- Bloquer la revue A32 jusqu'a l'execution des quatre followups.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_handoff_compiler.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_handoff_compiler.py`
- `diagnostics/sage/sage6d_second_unknown_game_handoff.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `source_candidate_handoff_frontiers=1`
- `handoff_items=1`
- `pre_registered_followup_protocols=4`
- `control_diversity_protocols=3`
- `control_diversity_budgets=[50,150,300]`
- `neutral_context_replication_protocols=1`
- `pre_registered_new_control_actions=[ACTION3]`
- `executed_distinct_control_actions=1`
- `projected_distinct_control_actions_after_execution=2`
- `context_clusters_preserved=6`
- `protocol_contexts=4`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=1`
- `ready_for_followup_execution=1`
- `ready_for_A32_review=0`
- `execution_performed=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_HANDOFF_COMPILED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6D`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Protocoles pre-enregistres :

| type | budget | cluster | step | comparaison | effet precedent |
|---|---:|---|---:|---|---:|
| diversite de controle | 50 | `002` | 48 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| diversite de controle | 150 | `003` | 132 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| diversite de controle | 300 | `005` | 24 | ACTION2 - ACTION3 | +32 contre ACTION1 |
| replication neutre | 50 | `001` | 12 | ACTION2 - ACTION1 | 0 |

ACTION3 est le premier controle suggere par les quatre requests sources qui ne
soit ni ACTION2 ni le controle ACTION1 deja execute. Son emploi commun aux trois
budgets isole la variation de contexte et de budget. Le protocole ne suppose pas
que le delta restera +32 : pour les controles ACTION3, un delta strictement
positif est pre-enregistre comme coherence candidate-only et un delta nul ou
negatif comme deviation candidate-only. Pour la replication neutre, zero est la
condition de replication et tout delta non nul est une deviation. Aucun de ces
outcomes ne produit automatiquement support, confirmation, refutation ou write.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_handoff_compiler `
  --source-sage6c diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json `
  --source-sage6b diagnostics\sage\sage6b_second_unknown_game_m3_execution.json `
  --out diagnostics\sage\sage6d_second_unknown_game_handoff.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py -q
```

L'execution exacte demandee ici est realisee par SAGE.6e ci-dessous.

SAGE.6d autorise maintenant a dire : SAGE sait transformer un motif stable avec
exception en plan de falsification multi-budget auditable et fixe avant
execution. Il ne faut pas dire : ACTION3 a confirme le motif, le contexte neutre
a ete replique, ou le dossier est deja pret pour une revue A32.

## SAGE.6e - Exact pre-registered followup execution

Objectif :

- Executer les quatre protocoles SAGE.6d dans leur ordre source, sans selection
  fondee sur les outcomes.
- Rejouer exactement chaque prefixe live pour les bras cible et controle, puis
  verifier les deux hashes de contexte.
- Interdire toute substitution de protocole, contexte, budget, cible ou
  controle.
- Executer ACTION2 contre ACTION3 dans un contexte positif stable de chacun des
  budgets 50, 150 et 300.
- Repliquer ACTION2 contre ACTION1 dans le contexte neutre du budget 50.
- Appliquer les conditions d'interpretation pre-enregistrees sans transformer
  coherence en confirmation ni deviation en refutation.
- Produire une assessment candidate-only de la dependance au controle avant
  toute consolidation ou revue A32.
- Garder `support=0`, la quarantaine A33.2 et l'absence de write A32/A33.

Ajouts :

- `theory/sage/second_unknown_game_followup_execution.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_followup_execution.py`
- `diagnostics/sage/sage6e_second_unknown_game_followup_execution.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `protocols_available=4`
- `protocols_selected=4`
- `protocols_executed=4`
- `protocols_blocked=0`
- `protocols_executed_by_budget={50:2,150:1,300:1}`
- `control_diversity_protocols_executed=3`
- `neutral_replication_protocols_executed=1`
- `live_prefix_replay_exact_events=4`
- `context_snapshot_hash_verified_events=4`
- `protocol_execution_exact_events=4`
- `target_actions_executed={ACTION2:4}`
- `control_actions_executed={ACTION1:1,ACTION3:3}`
- `target_signal_total=130`
- `control_signal_total=130`
- `controlled_effect_sizes=[0,0,0,0]`
- `positive_effect_events=0`
- `negative_effect_events=0`
- `zero_effect_events=4`
- `raw_support_events=0`
- `raw_contradiction_events=0`
- `raw_neutral_events=4`
- `pre_registered_conditions_met=1`
- `pre_registered_deviations=3`
- `distinct_control_matches_target_across_all_budgets=true`
- `neutral_context_replication_matched=true`
- `ready_for_post_execution_consolidation=true`
- `ready_for_A32_review=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_FOLLOWUPS_EXECUTED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6E`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Resultats pre-enregistres :

| type | budget | cluster | comparaison | cible | controle | delta | interpretation |
|---|---:|---|---|---:|---:|---:|---|
| diversite | 50 | `002` | ACTION2 - ACTION3 | 33 | 33 | 0 | deviation candidate-only |
| diversite | 150 | `003` | ACTION2 - ACTION3 | 33 | 33 | 0 | deviation candidate-only |
| diversite | 300 | `005` | ACTION2 - ACTION3 | 32 | 32 | 0 | deviation candidate-only |
| replication | 50 | `001` | ACTION2 - ACTION1 | 32 | 32 | 0 | condition satisfaite candidate-only |

Le controle ACTION3 egale ACTION2 sur les trois budgets et annule ainsi les
deltas +32 observes contre ACTION1 dans les memes familles de contextes. Cela
indique un motif candidat sensible a l'identite du controle, pas une mecanique
ACTION2 isolee. La replication exacte du contexte neutre reste a zero. Les trois
deltas nuls ACTION2/ACTION3 sont des deviations par rapport a la condition
positive pre-enregistree; ils ne sont ni contradictions scientifiques ni
refutations. La replication neutre satisfaite n'est pas une confirmation.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_followup_execution `
  --source-sage6d diagnostics\sage\sage6d_second_unknown_game_handoff.json `
  --out diagnostics\sage\sage6e_second_unknown_game_followup_execution.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_followup_execution.py `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py `
  tests\test_sage_second_unknown_game_m3_execution.py -q
```

La consolidation et la reevaluation demandees ici sont realisees par SAGE.6f
ci-dessous.

SAGE.6e autorise maintenant a dire : SAGE sait executer un plan de falsification
pre-enregistre et detecter qu'un effet apparent depend du controle choisi. Il ne
faut pas dire : ACTION2 est sans effet, ACTION1 ou ACTION3 definit la bonne
baseline scientifique, ou le dossier est deja eligible a une revision A32.

## SAGE.6f - Control-dependence consolidation and A32 eligibility

Objectif :

- Consolider les six observations ACTION2/ACTION1 de SAGE.6c avec les quatre
  followups SAGE.6e, sans nouvelle execution.
- Conserver exactement les six clusters de contexte SAGE.6c et rattacher la
  replication neutre a son contexte d'origine sans la compter comme contexte
  independant.
- Garder les identites ACTION1 et ACTION3 separees dans les groupes d'effet.
- Construire trois comparaisons appairees ACTION1/ACTION3, une par budget.
- Reformuler le candidat comme contraste local dependant du controle, jamais
  comme effet ACTION2 autonome et inconditionnel.
- Evaluer l'eligibilite a une revue A32 avec des criteres explicites : volume
  brut, deux controles, trois contextes appaires, couverture multi-budget,
  replication neutre, absence d'effet negatif et preservation des contextes.
- Emettre un frontier A32 candidate-only si tous les criteres passent, sans
  demander l'intake ni ecrire A32/A33.
- Garder `support=0`, la quarantaine A33.2 et l'absence de verdict SAGE.

Ajouts :

- `theory/sage/second_unknown_game_control_dependence_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_second_unknown_game_control_dependence_consolidation.py`
- `diagnostics/sage/sage6f_second_unknown_game_control_dependence_consolidation.json`

Run du 2026-07-19 :

- `game_id=wa30-ee6fef47`
- `budgets=[50,150,300]`
- `observation_records=10`
- `source_sage6c_observations=6`
- `source_sage6e_observations=4`
- `context_clusters=6`
- `context_clusters_preserved=6`
- `cross_context_merges_performed=0`
- `control_actions=[ACTION1,ACTION3]`
- `distinct_control_actions=2`
- `control_conditioned_effect_groups=3`
- `paired_control_contexts=3`
- `paired_control_budgets=[50,150,300]`
- `paired_control_effect_gaps=[32,32,32]`
- `paired_control_effect_gap_spread=0`
- `action1_positive_contexts=5`
- `action1_neutral_observations=2`
- `action3_neutral_contexts=3`
- `replicated_neutral_contexts=1`
- `negative_effect_events=0`
- `raw_support_events=5`
- `raw_contradiction_events=0`
- `raw_neutral_events=5`
- `eligibility_requirements_passed=14`
- `eligibility_requirements_total=14`
- `missing_eligibility_requirements=[]`
- `candidate_a32_review_frontiers=1`
- `ready_for_A32_review=1`
- `ready_for_A32_review_is_not_verdict=true`
- `a32_intake_requested=false`
- `gate_passed=true`
- `outcome_status=SAGE_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_CONSOLIDATED_A32_REVIEW_ELIGIBLE_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_6F`
- `revision_status=CANDIDATE_ONLY`
- `a32_write_performed=false`
- `a33_write_performed=false`

Groupes conditionnes par controle :

| controle | direction | observations | contextes independants | budgets | effets |
|---|---|---:|---:|---|---|
| ACTION1 | positive | 5 | 5 | 50/150/300 | 32/32/32/32/32 |
| ACTION1 | neutre | 2 | 1 | 50 | 0/0 |
| ACTION3 | neutre | 3 | 3 | 50/150/300 | 0/0/0 |

Clusters consolides :

| cluster | budget | step | observations | controles | statut candidate-only |
|---|---:|---:|---:|---|---|
| `001` | 50 | 12 | 2 | ACTION1 | neutre replique |
| `002` | 50 | 48 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `003` | 150 | 132 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `004` | 150 | 144 | 1 | ACTION1 | positif non apparie |
| `005` | 300 | 24 | 2 | ACTION1/ACTION3 | contraste dependant du controle |
| `006` | 300 | 36 | 1 | ACTION1 | positif non apparie |

Dans chacun des trois contextes appaires, le signal cible est reproduit entre
les deux experiences, le delta ACTION2/ACTION1 vaut 32 et le delta
ACTION2/ACTION3 vaut 0. Le gap conditionne par controle vaut donc 32 avec une
dispersion nulle sur les trois budgets. La replication du cluster `001` reste
neutre et n'ajoute pas artificiellement un contexte independant.

Les quatorze criteres d'eligibilite passent. Le frontier demande a A32 de revoir
la proposition suivante : le contraste local ACTION2 est positif contre ACTION1
mais neutre contre ACTION3 dans les contextes wa30 appaires. Il interdit une
revue sous la forme `STANDALONE_UNCONDITIONAL_ACTION2_EFFECT`. Cette eligibilite
n'est ni une confirmation, ni une revision, ni une demande d'intake automatique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.second_unknown_game_control_dependence_consolidation `
  --source-sage6c diagnostics\sage\sage6c_second_unknown_game_event_consolidation.json `
  --source-sage6e diagnostics\sage\sage6e_second_unknown_game_followup_execution.json `
  --out diagnostics\sage\sage6f_second_unknown_game_control_dependence_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_second_unknown_game_control_dependence_consolidation.py `
  tests\test_sage_second_unknown_game_followup_execution.py `
  tests\test_sage_second_unknown_game_handoff_compiler.py `
  tests\test_sage_second_unknown_game_event_consolidation.py -q
```

Suite conseillee apres SAGE.6f :

1. A32.6 - effectuer la revue scientifique du candidat dependant du controle,
   en decidant notamment si ACTION1 est un comparateur valide et si
   l'equivalence ACTION3 rend la revendication ACTION2 autonome non identifiable.
2. Ne soumettre a A33 qu'une eventuelle decision confirmee et strictement limitee
   a `wa30-ee6fef47` et aux conditions de controle/contextes revues.
3. Conserver `tn36` comme troisieme jeu inconnu pre-enregistre, sans le choisir
   sur la base de son outcome.

SAGE.6f autorise maintenant a dire : SAGE sait consolider des observations
multi-controles sans confondre baseline, contexte et support scientifique, puis
preparer un dossier A32 correctement reformule. Il ne faut pas dire : ACTION2 a
une mecanique confirmee, ACTION1 est la baseline correcte, ou le frontier A32 est
deja une decision scientifique.

## SAGE.7 - Third unknown-game bounded transfer

Objectif :

- Fermer la boucle SAGE.6f -> A32.6 -> A33.3 avant de changer de jeu.
- Reprendre l'ordre `public_unseen` fixe sans lire aucune metrique d'outcome.
- Exclure les deux jeux inconnus deja utilises et les scopes enregistres :
  `sb26-7fbdac44` et `wa30-ee6fef47`.
- Selectionner `tn36-ab4f63cc`, prochain jeu inconnu eligible pre-enregistre.
- Rejouer la discipline bornee sur budgets 50, 150 et 300.
- Lire A33.2 et A33.3 uniquement comme gardes de quarantaine ; ne reutiliser ni
  ACTION5/sb26, ni la relation ACTION2/ACTION1/ACTION3 de wa30.
- Auditer la surface d'action live au RESET et conserver les variantes
  parametrees ACTION6 comme variantes d'une seule famille d'action.
- Garder le resultat candidate-only, `support=0`, sans intake ni write A32/A33.

Ajouts :

- `theory/sage/third_unknown_game_transfer.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_transfer.py`
- `diagnostics/sage/sage7_third_unknown_game_transfer_results.json`

Run du 2026-07-19 :

- `first_unknown_game_id=sb26-7fbdac44`
- `second_unknown_game_id=wa30-ee6fef47`
- `selected_third_game_id=tn36-ab4f63cc`
- `candidate_games_audited=5`
- `eligible_unknown_games=1`
- `known_or_seen_candidates_excluded=2`
- `prior_or_registry_scope_candidates_excluded=2`
- `outcome_metrics_read_for_selection=false`
- `source_a33_2_entries_quarantined=1`
- `source_a33_3_entries_quarantined=1`
- `budgets_evaluated=[50,150,300]`
- `budgets_gate_passed=3`
- `all_budgets_gate_passed=true`
- `env_steps_total=172`
- `subgoal_switches_total=172`
- `new_candidate_targets_discovered_total=27`
- `rerun_m2_m3_requested_total=145`
- `rerun_m2_m3_effective_requests_generated_total=0`
- `budgets_with_progress_stall_detected=[]`
- `max_terminal_rate=0.016393`
- `max_repeated_action_arg_rate=0`
- `min_unique_state_signatures=51`
- `max_unique_state_signatures=62`
- `levels_completed_max=0`
- `action_families=[ACTION6]`
- `distinct_action_families=1`
- `legal_action_options_count=11`
- `parameterized_action_options_count=11`
- `parameterized_action_variants_counted_as_distinct_actions=false`
- `ready_for_parameterized_mini_frontier=true`
- `required_next_step=PARAMETERIZED_ACTION6_MINI_FRONTIER_REQUIRED_CANDIDATE_ONLY`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Selection pre-execution :

| rang fixe | jeu | decision |
|---:|---|---|
| 1 | `wa30-ee6fef47` | exclu, jeu inconnu deja utilise et scope A33.3 |
| 2 | `tn36-ab4f63cc` | eligible, selectionne avant execution |
| 3 | `ft09-0d8bbf25` | exclu connu |
| 4 | `cn04-65d47d14` | exclu connu |
| 5 | `sb26-7fbdac44` | exclu, jeu inconnu deja utilise et scope A33.2 |

Surface d'action live au RESET :

- une seule famille : `ACTION6` ;
- onze options legales parametrees par `x/y` ;
- aucune variante n'est recomptee comme famille d'action distincte ;
- cette surface autorise une mini-frontiere parametree, pas une conclusion sur
  la mecanique du jeu.

Lecture :

- La discipline de boucle transfere techniquement sur un troisieme jeu inconnu :
  toutes les actions sont legales, aucune repetition catastrophique n'apparait
  et les taux terminaux restent sous le seuil.
- Le trigger progress-stall ne se declenche pas : chaque action change l'etat et
  la boucle alterne entre exploration de cibles ACTION6 et placeholders M2/M3.
- Les 145 demandes `rerun_m2_m3` restent des placeholders, avec zero requete
  effective. SAGE.7 n'est donc pas une boucle d'hypotheses complete.
- Les onze options ACTION6 montrent qu'une diversite parametree est disponible
  des le RESET. SAGE.7a doit la transformer en mini-frontieres live-prefix
  multi-budget sans feindre une diversite de familles d'action.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_transfer `
  --source-sage6 diagnostics\sage\sage6_second_unknown_game_transfer_results.json `
  --source-a33-2 diagnostics\a33\scoped_unknown_game_registry.json `
  --source-a33-3 diagnostics\a33\control_dependent_relational_registry.json `
  --out diagnostics\sage\sage7_third_unknown_game_transfer_results.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_transfer.py `
  tests\test_sage_second_unknown_game_transfer.py `
  tests\test_a33_control_dependent_relational_registry.py -q
```

Suite requise apres SAGE.7 :

1. SAGE.7a - materialiser une mini-frontiere ACTION6 parametree a partir des
   placeholders, avec contextes live-prefix et budgets distribues.
2. Ne compter aucune variante parametree comme action distincte et conserver
   `support=0`.
3. N'ouvrir A32.7/A33.4 que si les executions controlees ulterieures produisent
   un dossier eligible selon des criteres fixes avant verdict.

SAGE.7 autorise maintenant a dire : SAGE transfere sa discipline bornee sur
trois jeux inconnus successifs et sait detecter qu'une surface mono-action exige
des controles parametres. Il ne faut pas dire : ACTION6 a une mecanique
confirmee, les onze coordonnees constituent onze actions distinctes, ou tn36 est
deja eligible a une revue A32.

## SAGE.7a - Third-game parameterized live mini-frontier

Objectif :

- Relire uniquement SAGE.7 comme contrat de transfert et de quarantaine.
- Reproduire les trois runs bornes de `tn36-ab4f63cc` avant tout replay de
  prefixe live.
- Attribuer les 172 switches et les 145 placeholders observes par SAGE.7.
- Selectionner six placeholders par budget selon des ordinaux espaces fixes,
  sans lire les outcomes.
- Transformer chaque placeholder selectionne en requete M3 live-prefix dont la
  cible et les deux controles appartiennent tous a ACTION6, avec des arguments
  `x/y` differents.
- Pre-enregistrer les controles live-legaux avant leur execution selon la regle
  `MAX_MANHATTAN_DISTANCE_FROM_TARGET_THEN_CANONICAL_ARGS`.
- Ne jamais recompter les variantes parametrees comme familles d'action.
- Conserver les hypotheses et requetes candidate-only, `support=0`, sans intake
  ni write A32/A33.

Ajouts :

- `theory/sage/third_unknown_game_parameterized_frontier.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_parameterized_frontier.py`
- `diagnostics/sage/sage7a_third_unknown_game_parameterized_frontier.json`

Run du 2026-07-19 :

- `game_id=tn36-ab4f63cc`
- `budgets_evaluated=[50,150,300]`
- `source_switches_expected=172`
- `switches_reproduced=172`
- `total_switch_events_observed=174`
- `terminal_guard_events_outside_source_switch_count=2`
- `true_exploratory_switches=27`
- `new_candidate_target_switches=27`
- `placeholder_rerun_m2_m3_switches=145`
- `requests_per_budget_target=6`
- `requests_by_budget={50:6,150:6,300:6}`
- `mini_frontier_hypotheses_generated=18`
- `effective_requests_generated=18`
- `parameterized_control_variants_pre_registered=36`
- `controls_per_request=2`
- `action_families=[ACTION6]`
- `distinct_action_families=1`
- `distinct_target_parameter_variants=3`
- `distinct_control_parameter_variants=4`
- `unique_context_snapshot_hashes=10`
- `parameterized_variants_counted_as_distinct_actions=false`
- `all_requests_ready_for_m3=true`
- `all_requests_live_prefix_replayable=true`
- `ready_for_parameterized_m3_execution=true`
- `required_next_step=SAGE7B_PARAMETERIZED_ACTION6_EXECUTION_REQUIRED_CANDIDATE_ONLY`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_PARAMETERIZED_FRONTIER_GENERATED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7A`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Distribution pre-enregistree :

| budget | placeholders | ordinaux selectionnes | requetes | controles |
|---:|---:|---|---:|---:|
| 50 | 41 | `0,8,16,24,32,40` | 6 | 12 |
| 150 | 52 | `0,10,20,30,40,51` | 6 | 12 |
| 300 | 52 | `0,10,20,30,40,51` | 6 | 12 |

Lecture :

- Le generateur generique SAGE.5c excluait une action portant le meme nom que
  la cible et bloquait donc toute surface mono-action. SAGE.7a remplace ce faux
  blocage par une identite experimentale `(ACTION6,x,y)` tout en conservant une
  seule famille ACTION6.
- Chaque controle est disponible dans le meme contexte live-prefix que sa cible,
  utilise ACTION6 et porte des arguments differents. Les variantes sont choisies
  avant toute execution M3 et ne sont pas du support.
- Les runs 150 et 300 atteignent les memes prefixes aux ordinaux communs. Les 18
  requetes couvrent donc 10 contextes uniques et fournissent volontairement des
  repetitions inter-budget, sans deduplication post-outcome.
- SAGE.7a ne mesure aucun effet cible/controle. Il materialise un plan
  experimental strictement rejouable ; il ne produit ni confirmation, ni
  refutation, ni dossier A32.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_parameterized_frontier `
  --source-sage7 diagnostics\sage\sage7_third_unknown_game_transfer_results.json `
  --out diagnostics\sage\sage7a_third_unknown_game_parameterized_frontier.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_parameterized_frontier.py `
  tests\test_sage_third_unknown_game_transfer.py -q
```

Suite requise apres SAGE.7a :

1. SAGE.7b - executer exactement les 18 cibles et leurs 36 controles
   pre-enregistres, dans les memes contextes live-prefix et sans substitution.
2. Conserver les mesures comme evenements candidate-only avec `support=0`.
3. Consolider ensuite les repetitions inter-budget avant de decider si un
   dossier A32.7 est identifiable, non identifiable ou encore incomplet.

SAGE.7a autorise maintenant a dire : SAGE sait construire une mini-frontiere
controlee sur une surface mono-action en separant famille d'action et arguments.
Il ne faut pas dire : une coordonnee ACTION6 est une nouvelle action, l'une des
variantes a un effet propre, ou le dossier est deja eligible a A32.7.

## SAGE.7b - Exact parameterized ACTION6 execution

Objectif :

- Consommer uniquement les 18 requetes pre-enregistrees par SAGE.7a.
- Executer chaque cible ACTION6 une fois et ses deux controles ACTION6 exacts,
  soit 54 bras live-prefix sans deduplication inter-budget.
- Verifier le hash du contexte avant chaque bras et interdire toute substitution
  d'action, de coordonnees ou de prefixe.
- Comparer chaque cible a chacun de ses controles avec la metrique inscrite dans
  la requete, sans transformer les deltas en support ou refutation.
- Preserver les repetitions entre budgets pour une consolidation ulterieure.
- Garder le resultat candidate-only, `support=0`, sans intake A32 ni write A33.

Ajouts :

- `theory/sage/third_unknown_game_parameterized_execution.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_parameterized_execution.py`
- `diagnostics/sage/sage7b_third_unknown_game_parameterized_execution.json`

Run du 2026-07-19 :

- `game_id=tn36-ab4f63cc`
- `budgets_evaluated=[50,150,300]`
- `requests_available=18`
- `requests_executed=18`
- `requests_blocked=0`
- `requests_executed_by_budget={50:6,150:6,300:6}`
- `target_arm_executions=18`
- `control_arm_executions=36`
- `total_arm_executions=54`
- `comparison_events=36`
- `live_prefix_replay_exact_events=18`
- `protocol_exact_match_events=18`
- `protocol_substitution_events=0`
- `metrics_executed={local_patch_before_after:32,terminal_state_after_rollout:4}`
- `positive_delta_events=13`
- `negative_delta_events=0`
- `zero_delta_events=23`
- `distinct_effect_sizes=[0.0,2.0]`
- `discrimination_statuses={DISCRIMINATING_TARGET_EFFECT_CANDIDATE_ONLY:13,NON_DISCRIMINATING_EQUAL_EFFECT_CANDIDATE_ONLY:23}`
- `action_families=[ACTION6]`
- `distinct_action_families=1`
- `parameterized_variants_counted_as_distinct_actions=false`
- `ready_for_event_consolidation=true`
- `required_next_step=SAGE7C_PARAMETERIZED_EVENT_CONSOLIDATION_REQUIRED_CANDIDATE_ONLY`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_PARAMETERIZED_EXECUTION_COMPLETED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7B`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Observations par budget :

| budget | requetes | bras cible | bras controle | deltas positifs | deltas negatifs | deltas nuls |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 6 | 6 | 12 | 5 | 0 | 7 |
| 150 | 6 | 6 | 12 | 4 | 0 | 8 |
| 300 | 6 | 6 | 12 | 4 | 0 | 8 |

Lecture :

- Pour la cible `ACTION6 {x:25,y:42}`, les treize comparaisons contre
  `ACTION6 {x:34,y:51}` donnent un signal cible `2`, un signal controle `0` et
  un delta `+2`.
- Les treize comparaisons de la meme cible contre
  `ACTION6 {x:41,y:44}` donnent `2` des deux cotes et un delta nul. L'effet
  observe depend donc du controle parametre ; il ne peut pas etre attribue a la
  seule cible ACTION6.
- Les six comparaisons du premier contexte, cible `{x:35,y:42}`, sont neutres.
  Les quatre comparaisons terminales, cible `{x:30,y:42}`, sont aussi neutres.
- Les runs 150 et 300 reproduisent exactement leurs contextes communs. Ces
  duplications sont conservees comme repetitions techniques mais ne doivent pas
  etre recomptees comme contextes independants.
- SAGE.7b fournit des evenements bruts mixtes. Il ne conclut ni mecanique
  autonome, ni non-identifiabilite globale, ni eligibilite A32.7.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_parameterized_execution `
  --source-sage7a diagnostics\sage\sage7a_third_unknown_game_parameterized_frontier.json `
  --out diagnostics\sage\sage7b_third_unknown_game_parameterized_execution.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_parameterized_execution.py `
  tests\test_sage_third_unknown_game_parameterized_frontier.py -q
```

Suite requise apres SAGE.7b :

1. SAGE.7c - consolider les 36 comparaisons par hash de contexte, cible,
   controle et metrique, en separant repetitions inter-budget et contextes
   independants.
2. Identifier si les deltas `+2` contre `{x:34,y:51}` restent strictement
   dependants du controle nul `{x:41,y:44}`.
3. N'ouvrir A32.7 que si la consolidation produit un dossier identifiable selon
   des criteres fixes, sans convertir automatiquement les 13 deltas positifs en
   support scientifique.

SAGE.7b autorise maintenant a dire : SAGE execute exactement une experience
parametree multi-controles sur une surface mono-action et observe un contraste
dependant du controle. Il ne faut pas dire : `{x:25,y:42}` a un effet autonome,
les 13 deltas sont 13 supports independants, ou A32.7 est deja autorise.

## SAGE.7c - Context-aware parameterized event consolidation

Objectif :

- Consommer uniquement les evenements exacts SAGE.7b.
- Consolider chaque comparaison par jeu, hash de contexte, cible, controle et
  metrique.
- Compter un hash de contexte comme un seul contexte independant, meme lorsqu'il
  est rejoue sous plusieurs budgets.
- Construire une evaluation multi-controles par contexte puis un dossier par
  variante cible et metrique.
- Fixer avant verdict les criteres d'eligibilite : au moins trois contextes
  independants, deux controles communs par contexte, au moins un contexte
  replique entre budgets, aucune inversion et des repetitions exactes coherentes.
- Separer le contraste relationnel dependant du controle de tout effet autonome
  de la cible.
- Produire seulement une eligibilite de handoff, jamais une decision A32.

Ajouts :

- `theory/sage/third_unknown_game_parameterized_consolidation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_parameterized_consolidation.py`
- `diagnostics/sage/sage7c_third_unknown_game_parameterized_consolidation.json`

Run du 2026-07-19 :

- `game_id=tn36-ab4f63cc`
- `raw_comparison_events=36`
- `consolidated_comparison_groups=20`
- `technical_replication_events=16`
- `independent_parameterized_contexts=10`
- `cross_budget_replicated_contexts=6`
- `control_dependent_contexts=8`
- `non_discriminating_contexts=2`
- `contradictory_contexts=0`
- `parameterized_candidate_dossiers=3`
- `a32_handoff_eligible_candidates=1`
- `eligible_candidate_target_args=[{x:25,y:42}]`
- `eligible_candidate_independent_contexts=8`
- `eligible_candidate_cross_budget_replicated_contexts=4`
- `eligible_candidate_raw_comparison_events=26`
- `eligible_candidate_technical_replication_events=10`
- `autonomous_target_effects_confirmed=0`
- `autonomous_target_effects_unresolved=3`
- `parameterized_variants_counted_as_distinct_actions=false`
- `ready_for_a32_handoff_compilation=true`
- `required_next_step=SAGE7D_A32_CONTROL_DEPENDENT_HANDOFF_REQUIRED_CANDIDATE_ONLY`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_CONTROL_DEPENDENT_PARAMETERIZED_DOSSIER_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7C`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Dossiers cibles :

| cible ACTION6 | metrique | contextes independants | statut contextuel | handoff A32 |
|---|---|---:|---|---|
| `{x:25,y:42}` | `local_patch_before_after` | 8 | 8 dependants du controle | eligible |
| `{x:35,y:42}` | `local_patch_before_after` | 1 | non discriminant | non |
| `{x:30,y:42}` | `terminal_state_after_rollout` | 1 | non discriminant | non |

Contraste eligible :

- cible : `ACTION6 {x:25,y:42}` ;
- controle discriminant commun aux huit contextes :
  `ACTION6 {x:34,y:51}`, delta `+2` ;
- comparateur equivalent commun aux huit contextes :
  `ACTION6 {x:41,y:44}`, delta `0` ;
- contextes independants : `8` ;
- contextes repliques entre budgets : `4` ;
- evenements bruts : `26`, dont `10` repetitions techniques ;
- effet autonome de la cible : `UNRESOLVED_CONTROL_DEPENDENT_TARGET_EFFECT` ;
- support scientifique compte par SAGE.7c : `0`.

Lecture :

- Les 13 deltas positifs SAGE.7b deviennent huit contrastes contextuels
  independants ; cinq sont des repetitions techniques. La meme reduction est
  appliquee aux evenements neutres.
- Dans chacun des huit contextes eligibles, la cible differe du controle
  `{x:34,y:51}` mais reste equivalente au controle `{x:41,y:44}`. Le dossier
  porte donc une relation cible/controles, pas un effet propre a la cible.
- Les deux controles sont communs a tous les contextes eligibles et leurs roles
  ne changent jamais. Quatre contextes sont reproduits sous plusieurs budgets.
- Cette regularite satisfait les criteres de handoff fixes, mais SAGE.7c ne
  transforme ni les huit contextes ni leurs repetitions en support scientifique.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_parameterized_consolidation `
  --source-sage7b diagnostics\sage\sage7b_third_unknown_game_parameterized_execution.json `
  --out diagnostics\sage\sage7c_third_unknown_game_parameterized_consolidation.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_parameterized_consolidation.py `
  tests\test_sage_third_unknown_game_parameterized_execution.py -q
```

Suite requise apres SAGE.7c :

1. SAGE.7d - compiler un handoff A32.7 portant uniquement le contraste
   relationnel `{x:25,y:42}` / `{x:34,y:51}` / `{x:41,y:44}`, les huit hashes
   exacts, les metriques et les identifiants source.
2. Ne pas inclure les cibles non discriminantes `{x:35,y:42}` et `{x:30,y:42}`.
3. A32.7 devra choisir scientifiquement entre confirmation relationnelle
   limitee au scope, non-identifiabilite ou demande d'experiences ; SAGE.7d ne
   doit programmer aucun de ces verdicts.

SAGE.7c autorise maintenant a dire : SAGE sait dedupliquer les repetitions
inter-budget et preparer un candidat relationnel dependant du controle dans huit
contextes independants. Il ne faut pas dire : huit supports sont confirmes,
`{x:25,y:42}` a un effet autonome, ou A32.7 a deja rendu son verdict.

## SAGE.7d - A32.7 relational handoff compiler

Objectif :

- Consommer uniquement SAGE.7c.
- Inclure le seul candidat marque `a32_handoff_eligible=true`.
- Verrouiller le jeu, la cible, les deux controles parametres, la metrique, les
  huit hashes de contexte et tous les liens vers les groupes consolides.
- Conserver les 26 evenements bruts et les 10 repetitions techniques sans les
  recompter comme contextes ou support.
- Exclure explicitement les deux cibles non discriminantes.
- Presenter a A32.7 les quatre decisions autorisees sans en preselectionner une.
- Garder l'effet autonome de la cible unresolved et ne produire aucun verdict.

Ajouts :

- `theory/sage/third_unknown_game_a32_handoff.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_third_unknown_game_a32_handoff.py`
- `diagnostics/sage/sage7d_third_unknown_game_a32_handoff.json`

Run du 2026-07-19 :

- `game_id=tn36-ab4f63cc`
- `source_candidate_dossiers=3`
- `source_a32_handoff_eligible_candidates=1`
- `handoff_items=1`
- `excluded_candidate_items=2`
- `handoff_contexts=8`
- `handoff_cross_budget_replicated_contexts=4`
- `handoff_raw_comparison_events=26`
- `handoff_technical_replication_events=10`
- `handoff_candidate_support_events=8`
- `a32_decision_preselected=false`
- `autonomous_target_effects_confirmed=0`
- `autonomous_target_effects_unresolved=1`
- `parameterized_variants_counted_as_distinct_actions=false`
- `ready_for_a32_7_scientific_review=true`
- `required_next_step=A32_7_CONTROL_DEPENDENT_PARAMETERIZED_SCIENTIFIC_REVIEW_REQUIRED`
- `gate_passed=true`
- `outcome_status=SAGE_THIRD_UNKNOWN_GAME_A32_HANDOFF_COMPILED_CANDIDATE_ONLY`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_SAGE_7D`
- `revision_status=CANDIDATE_ONLY`
- `a32_intake_requested=false`
- `a32_write_performed=false`
- `a33_write_performed=false`

Handoff A32.7 :

- cible : `ACTION6 {x:25,y:42}` ;
- controle discriminant : `ACTION6 {x:34,y:51}` ;
- comparateur equivalent : `ACTION6 {x:41,y:44}` ;
- metrique : `local_patch_before_after` ;
- scope : `EXACT_TN36_LIVE_PREFIX_CONTEXTS_AND_PARAMETER_VARIANTS` ;
- huit contextes independants, dont quatre reproduits entre budgets ;
- `candidate_support_events=8`, tous non comptes comme support scientifique ;
- `autonomous_target_effect_status=UNRESOLVED_CONTROL_DEPENDENT_TARGET_EFFECT` ;
- `handoff_status=READY_FOR_A32_7_SCIENTIFIC_REVIEW_CANDIDATE_ONLY`.

Decisions A32.7 autorisees, non preselectionnees :

1. `CONFIRM_SCOPE_LIMITED_CONTROL_DEPENDENT_PARAMETERIZED_RELATION` ;
2. `KEEP_UNRESOLVED_NON_IDENTIFIABLE_PARAMETERIZED_TARGET_EFFECT` ;
3. `REQUEST_MORE_TESTS_FOR_PARAMETERIZED_RELATION` ;
4. `REFUTE_CONTROL_DEPENDENT_PARAMETERIZED_RELATION`.

Exclusions :

- `ACTION6 {x:35,y:42}` / `local_patch_before_after` : non discriminant ;
- `ACTION6 {x:30,y:42}` / `terminal_state_after_rollout` : non discriminant.

Lecture :

- SAGE.7d preserve la topologie complete du contraste : chaque contexte porte
  le signal cible `2`, le controle discriminant `0` et le comparateur equivalent
  `2`, avec les identifiants de comparaison correspondants.
- Le handoff ne demande pas a A32.7 de confirmer la cible autonome. Il lui
  demande d'evaluer une relation precise et de conserver explicitement la
  possibilite de non-identifiabilite ou de nouvelles experiences.
- Les deux candidats non discriminants ne sont ni refutes ni transferes ; ils
  restent simplement hors du scope de la revue.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage.third_unknown_game_a32_handoff `
  --source-sage7c diagnostics\sage\sage7c_third_unknown_game_parameterized_consolidation.json `
  --out diagnostics\sage\sage7d_third_unknown_game_a32_handoff.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_sage_third_unknown_game_a32_handoff.py `
  tests\test_sage_third_unknown_game_parameterized_consolidation.py -q
```

Suite requise apres SAGE.7d :

1. A32.7 - effectuer la revue scientifique du contraste relationnel dans son
   scope exact, sans recompter les repetitions techniques.
2. Decider separement le statut de la relation cible/controles et celui de
   l'effet autonome de `{x:25,y:42}`.
3. Ne produire un handoff A33.4 que si A32.7 emet une confirmation relationnelle
   scope-limited avec support scientifique explicitement compte par A32.

SAGE.7d autorise maintenant a dire : le dossier tn36 est pret pour une revue
A32.7 exacte. Il ne faut pas dire : A32.7 a confirme la relation, les huit
contextes sont deja du support scientifique, ou A33.4 est autorise.

## SAGE.8a - relational memory policy integration

Objectif :

- Consommer les relations scope-limited confirmees dans A33.3 et A33.4.
- Compiler une politique executable qui remplace uniquement le comparateur de
  moindre effet dans le jeu, la metrique et les hashes de contexte exacts.
- Preserver les comparateurs equivalents et toute proposition hors scope.
- Exiger que l'action memoire choisie soit legalement disponible au moment de
  la decision.
- Ne pas reevaluer la verite, recompter le support ou supposer un effet autonome.

Ajouts :

- `theory/sage/relational_memory_policy.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_relational_memory_policy.py`
- `diagnostics/sage/sage8a_relational_memory_policy.json`

Run du 2026-07-19 :

- `registry_entries_consumed=2`
- `policy_entries_compiled=2`
- `games_scoped=[tn36-ab4f63cc, wa30-ee6fef47]`
- `exact_context_hashes_scoped=11`
- `exact_application_audits=2`
- `equivalent_comparators_preserved=2`
- `wrong_context_overrides=0`
- `wrong_game_overrides=0`
- `registry_truth_reevaluations=0`
- `registry_support_recounted=0`
- `comparative_evaluation_performed=false`
- `ready_for_comparative_evaluation=true`
- `gate_passed=true`
- `outcome_status=SAGE_RELATIONAL_MEMORY_POLICY_READY_FOR_COMPARISON`

Lecture : SAGE sait maintenant consulter ses deux memoires relationnelles pour
modifier une decision concrete, mais uniquement dans les onze contextes
enregistres. SAGE.8a prouve l'integration de la politique, pas encore un gain de
niveau ou de victoire.

Suite requise : SAGE.8b doit comparer avec et sans memoire sur les memes replays,
en prenant `levels_completed` et `win_rate` comme metriques principales et le
signal local uniquement comme metrique secondaire.

## SAGE.8b - relational memory paired A/B evaluation

Objectif :

- Rejouer les onze contextes scope-locked avec un bras sans memoire et un bras
  utilisant la decision executable de SAGE.8a.
- Utiliser exactement le meme prefixe dans les deux bras et verifier le hash de
  contexte avant l'action evaluee.
- Mesurer en premier `levels_completed` et `win_rate`.
- Garder `local_patch_before_after` comme diagnostic secondaire, sans le
  convertir en victoire ou en niveau termine.

Ajouts :

- `theory/sage/relational_memory_ab_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_relational_memory_ab_evaluation.py`
- `diagnostics/sage/sage8b_relational_memory_ab_evaluation.json`

Run du 2026-07-19 :

- `paired_episodes_evaluated=11`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `memory_policy_applications=11`
- `exact_paired_replays=11`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `levels_completed_improved=false`
- `no_memory_wins=0`
- `with_memory_wins=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `win_rate_improved=false`
- `secondary_local_signal_gain=112.0`
- `secondary_local_signal_improved=true`
- `primary_arc_progress_improved=false`
- `local_signal_counted_as_arc_progress=false`
- `gate_passed=true`
- `outcome_status=SAGE_RELATIONAL_MEMORY_LOCAL_GAIN_WITHOUT_ARC_SCORE_GAIN`

Lecture : la memoire relationnelle influence correctement les onze decisions et
ameliore le signal local dans chacune d'elles (+96 sur wa30, +16 sur tn36).
Cependant, dans cet horizon d'une action apres le prefixe, elle n'augmente ni le
nombre de niveaux termines ni le taux de victoire. Le resultat est donc un gain
fonctionnel local prouve, mais pas encore un gain ARC-AGI-3 primaire.

Suite requise : prolonger les traitements apres la decision memoire sur un
horizon multi-action et evaluer si le gain local se convertit en progression de
niveau ou en victoire, sans reutiliser les outcomes pour choisir les actions.

## SAGE.8c - relational memory multi-action conversion evaluation

Objectif :

- Prolonger les onze paires de SAGE.8b de 16 actions apres la decision initiale.
- Construire avant execution une continuation commune aux deux bras a partir
  des 16 dernieres actions du prefixe, sans lire aucun outcome futur.
- Ne faire diverger les bras que sur la decision initiale : comparateur de
  moindre effet sans memoire, choix SAGE.8a avec memoire.
- Arreter avant l'horizon uniquement si un etat terminal est atteint.
- Evaluer en premier `levels_completed` et `win_rate`, le changement local
  initial restant strictement secondaire.

Ajouts :

- `theory/sage/relational_memory_multi_action_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_relational_memory_multi_action_evaluation.py`
- `diagnostics/sage/sage8c_relational_memory_multi_action_evaluation.json`

Run du 2026-07-19 :

- `paired_rollouts_evaluated=11`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `continuation_horizon=16`
- `memory_policy_applications=11`
- `exact_paired_replays=11`
- `continuation_steps_requested_per_arm=176`
- `no_memory_continuation_steps_executed=171`
- `with_memory_continuation_steps_executed=171`
- `no_memory_terminal_episodes=1`
- `with_memory_terminal_episodes=1`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `no_memory_wins=0`
- `with_memory_wins=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `secondary_initial_local_signal_gain=112.0`
- `primary_arc_progress_improved=false`
- `primary_arc_progress_regressed=false`
- `continuation_selected_from_outcomes=false`
- `gate_passed=true`
- `outcome_status=SAGE_RELATIONAL_MEMORY_MULTI_ACTION_LOCAL_GAIN_WITHOUT_ARC_SCORE_CONVERSION`

Lecture : le gain local produit par la memoire survit comme difference initiale,
mais ne se convertit pas en niveau ou victoire sous cette continuation fixe. Les
deux bras executent 171 actions legales ; le seul arret anticipe est le meme
`GAME_OVER` tn36 dans les deux bras apres 11 actions de continuation. La memoire
n'aide ni ne degrade donc les metriques ARC principales dans ce protocole.

Suite requise : SAGE.8d doit remplacer la continuation historique fixe par une
politique fermee qui replanifie a partir de l'etat courant, tout en gardant une
selection aveugle aux outcomes et une evaluation avec/sans memoire appariee.

## SAGE.8d - relational memory state-conditioned closed-loop evaluation

Objectif :

- Remplacer la continuation fixe de SAGE.8c par un replanning apres chaque pas.
- Donner aux deux bras le meme algorithme et le meme horizon de 16 actions.
- Selectionner uniquement parmi les actions legales, a partir du digest visuel
  courant, des comptes d'actions passes et des visites etat/action.
- Compter toutes les variantes parametrees ACTION6 dans une seule famille.
- Exclure du planner les outcomes futurs, les niveaux, victoires et rollouts
  contrefactuels ; ils ne servent qu'a l'evaluation apres execution.
- Autoriser les trajectoires a diverger apres l'intervention memoire initiale.

Politique fermee :

1. privilegier une action non encore essayee dans l'etat visuel courant ;
2. eviter la repetition immediate ;
3. privilegier la famille d'action la moins utilisee ;
4. privilegier la variante concrete la moins utilisee ;
5. departager deterministement avec le digest visuel courant.

Ajouts :

- `theory/sage/relational_memory_closed_loop_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_relational_memory_closed_loop_evaluation.py`
- `diagnostics/sage/sage8d_relational_memory_closed_loop_evaluation.json`

Run du 2026-07-19 :

- `paired_rollouts_evaluated=11`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `continuation_horizon=16`
- `memory_policy_applications=11`
- `exact_paired_replays=11`
- `no_memory_replanning_decisions=171`
- `with_memory_replanning_decisions=171`
- `episodes_with_divergent_replanned_trajectories=11`
- `divergent_replanning_positions=119`
- `no_memory_terminal_episodes=1`
- `with_memory_terminal_episodes=1`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `no_memory_wins=0`
- `with_memory_wins=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `secondary_initial_local_signal_gain=112.0`
- `primary_arc_progress_improved=false`
- `primary_arc_progress_regressed=false`
- `future_outcomes_used_for_planning=false`
- `counterfactual_rollouts_performed=0`
- `gate_passed=true`
- `outcome_status=SAGE_RELATIONAL_MEMORY_CLOSED_LOOP_LOCAL_GAIN_WITHOUT_ARC_SCORE_CONVERSION`

Lecture : SAGE.8d prouve que SAGE peut replanifier en ligne a partir de l'etat
courant. Les onze paires divergent apres la decision initiale, sur 119 positions
de replanning, tout en restant legales et en utilisant exactement le meme
algorithme. Cette adaptativite ne convertit cependant toujours pas le gain local
en niveau ou victoire : l'exploration generique manque d'un objectif de jeu.

Suite requise : SAGE.8e doit ajouter au planner un objectif de progression
appris et scope-safe, sans optimiser sur les outcomes de ces onze episodes, puis
repeter l'evaluation appariee sur `levels_completed` et `win_rate`.

## SAGE.8e - scope-safe learned-objective closed-loop evaluation

Objectif :

- Remplacer l'exploration generique de SAGE.8d par un objectif de progression
  appris avant l'evaluation.
- Apprendre seulement dans les historiques de replay exacts A34.2 et A34.3 la
  prochaine action des prefixes qui atteignent les contextes relationnels
  enregistres.
- Limiter chaque prediction a la cle exacte `game_id|visual_digest` : aucune
  correspondance floue et aucun transfert inter-jeux.
- Mettre en quarantaine toute prediction hors scope ou devenue illegale, puis
  revenir au planner state-conditioned de SAGE.8d.
- Ne lire aucun outcome SAGE.8b, SAGE.8c, SAGE.8d ou SAGE.8e pour apprendre,
  regler ou choisir une action.
- Garder `levels_completed` et `win_rate` comme seules metriques ARC principales.

Le modele appris est explicitement un proxy de joignabilite des contextes
enregistres, pas une verite de progression ARC. Il contient 456 transitions de
demonstration reparties sur 137 etats visuels exacts, dont 24 ambigus, a partir
des onze replays verifies. Les deux bras emploient ensuite le meme planner et le
meme horizon de 16 actions ; seule l'intervention memoire initiale differe.

Ajouts :

- `theory/sage/relational_memory_objective_closed_loop_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_relational_memory_objective_closed_loop_evaluation.py`
- `diagnostics/sage/sage8e_relational_memory_objective_closed_loop_evaluation.json`

Run du 2026-07-19 :

- `paired_rollouts_evaluated=11`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `continuation_horizon=16`
- `memory_policy_applications=11`
- `exact_paired_replays=11`
- `objective_model_demonstration_transitions=456`
- `objective_model_exact_visual_states=137`
- `no_memory_replanning_decisions=171`
- `with_memory_replanning_decisions=171`
- `no_memory_objective_applications=28`
- `with_memory_objective_applications=112`
- `no_memory_objective_coverage_rate=0.16374269005847952`
- `with_memory_objective_coverage_rate=0.6549707602339181`
- `no_memory_objective_quarantines=143`
- `with_memory_objective_quarantines=59`
- `episodes_with_divergent_replanned_trajectories=10`
- `divergent_replanning_positions=134`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `no_memory_wins=0`
- `with_memory_wins=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `secondary_initial_local_signal_gain=112.0`
- `training_or_tuning_used_evaluation_outcomes=false`
- `future_outcomes_used_for_planning=false`
- `counterfactual_rollouts_performed=0`
- `scope_generalization_performed=false`
- `primary_arc_progress_improved=false`
- `primary_arc_progress_regressed=false`
- `gate_passed=true`
- `outcome_status=SAGE_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_LOCAL_GAIN_WITHOUT_ARC_SCORE_CONVERSION`

Lecture : l'intervention memoire maintient la trajectoire dans le manifold exact
du proxy appris beaucoup plus longtemps : 112 decisions couvertes contre 28,
soit 65,5 % contre 16,4 %. C'est un gain structurel net par rapport a SAGE.8d,
mais il ne se convertit toujours pas en niveau ou victoire. Le proxy reproduit
fidelement des prefixes connus ; il ne connait pas encore l'objectif reel du jeu.

Suite requise : SAGE.8f doit acquerir ou construire un signal d'objectif relie a
une transition de niveau ou de victoire, sans requalifier le proxy SAGE.8e et
sans regler le planner sur les outcomes de cette evaluation.

## SAGE.8f - goal-grounded signal acquisition and target admission

Objectif :

- Remplacer le proxy de joignabilite SAGE.8e par une banque de signaux dont
  chaque exemple est relie a une hausse observee de `levels_completed` ou a
  l'etat terminal `WIN`.
- Deriver le niveau avant l'action a partir du pas precedent du meme episode et
  verifier l'egalite exacte entre sa frame apres et la frame avant courante.
- Exclure les RESET, les transitions sans progression, les lignes sans
  predecesseur et toute rupture de continuite.
- Indexer chaque signal par la cle exacte `game_id|visual_digest`, sans fuzzy
  matching ni transfert inter-jeux.
- Auditer separement les 1 788 transitions preexistantes de `wa30` et `tn36`.
- N'autoriser le planner et une nouvelle evaluation fermee que si un signal
  positif existe dans chacun des jeux cibles eux-memes.
- Ne jamais utiliser les outcomes de l'evaluation SAGE.8e comme donnees
  d'apprentissage ou de tuning.

Cette etape distingue explicitement l'acquisition d'un signal objectif reel de
son admissibilite sur le domaine SAGE.8. Le partage d'un nom d'action entre deux
jeux n'autorise aucune reutilisation : sans exemple positif du meme jeu, le
signal reste en quarantaine.

Ajouts :

- `theory/sage/goal_grounded_signal_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_goal_grounded_signal_acquisition.py`
- `diagnostics/sage/sage8f_goal_grounded_signal_acquisition.json`

Run du 2026-07-19 :

- `source_trace_files=18`
- `source_rows_scanned=5711`
- `source_games_count=6`
- `frame_continuity_checks=5661`
- `frame_continuity_mismatches=0`
- `reset_rows_excluded=50`
- `non_goal_rows_excluded=5587`
- `verified_goal_transitions=74`
- `verified_level_up_transitions=74`
- `verified_win_transitions=5`
- `exact_goal_states=67`
- `ambiguous_exact_goal_states=3`
- `target_games_audited=2`
- `target_transitions_scanned=1788`
- `observed_target_goal_transitions=0`
- `exact_target_goal_signal_entries=0`
- `source_goal_signal_demonstrations_quarantined_from_transfer=74`
- `cross_game_transfer_performed=false`
- `planner_activation_authorized=false`
- `closed_loop_evaluation_performed=false`
- `evaluation_episodes_executed=0`
- `sage8e_evaluation_outcomes_used_for_training_or_tuning=false`
- `gate_passed=true`
- `outcome_status=SAGE_GOAL_GROUNDED_SIGNAL_ACQUIRED_TARGET_DOMAIN_COVERAGE_BLOCKED`

Lecture : SAGE possede desormais un signal d'objectif ancre dans de vraies
transitions de niveau et cinq victoires, avec continuite exacte. Cependant, les
six jeux sources ne comprennent ni `wa30` ni `tn36`. Leurs 1 788 transitions de
collecte ne contiennent elles-memes aucun level-up ou WIN. Appliquer les 74
demonstrations par simple analogie d'action serait donc une generalisation non
validee. SAGE.8f bloque volontairement le planner au lieu de produire une
nouvelle evaluation sans fondement cible.

Suite requise : SAGE.8g doit acquerir au moins une demonstration positive
rejouable dans chacun de `wa30` et `tn36`, ou produire un protocole actif borne
capable d'en trouver une sans lire le code des jeux. L'evaluation appariee ne
reprendra qu'apres admission exacte de ce signal cible.

## SAGE.8g - bounded active target-signal acquisition

Objectif :

- Acquerir une vraie transition de niveau dans chacun des deux jeux cibles a
  partir de l'API d'observation et d'actions legales uniquement.
- Ne lire aucun fichier source des jeux et ne reutiliser aucune demonstration
  provenant d'un autre `game_id`.
- Borner explicitement chaque protocole d'acquisition et n'utiliser
  `levels_completed` ou `game_state` que comme condition d'arret positive.
- Conserver la sequence complete depuis RESET et l'action positive finale.
- Rejouer chaque sequence sur une nouvelle instance et exiger l'egalite exacte
  des digests RESET, pre-action et post-action ainsi que de l'outcome.
- Indexer les signaux admis par `game_id|visual_digest`, sans fuzzy matching.
- Autoriser la prochaine evaluation seulement lorsque les deux jeux sont
  couverts ; ne pas lancer cette evaluation dans SAGE.8g.

Protocoles black-box :

- `tn36` : calibration des onze actions legales par delta de pixels, detection
  unique du bouton de validation par le plus petit delta, puis enumeration des
  1 024 configurations des dix commutateurs.
- `wa30` : replay borne de la trajectoire de transport obtenue par exploration
  des objets saillants visibles ; les trois objets 4x4 sont places dans le
  receptacle visible 12x4 en 50 actions.

Ajouts :

- `theory/sage/target_goal_signal_active_acquisition.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_target_goal_signal_active_acquisition.py`
- `diagnostics/sage/sage8g_target_goal_signal_active_acquisition.json`

Run du 2026-07-19 :

- `target_games_audited=2`
- `target_games_with_positive_transition=2`
- `target_games_with_exact_positive_replay=2`
- `verified_target_goal_transitions=2`
- `verified_target_level_up_transitions=2`
- `verified_target_win_transitions=0`
- `exact_target_goal_states=2`
- `exact_target_goal_signal_entries=2`
- `tn36_toggle_configuration_space=1024`
- `tn36_toggle_configurations_tested=859`
- `tn36_positive_configuration_mask=858`
- `tn36_positive_sequence_length=7`
- `wa30_positive_sequence_length=50`
- `discovery_action_executions=4965`
- `independent_replay_action_executions=57`
- `total_live_action_executions=5022`
- `game_source_files_opened=0`
- `future_outcomes_used_for_action_ranking=false`
- `cross_game_transfer_performed=false`
- `planner_activation_authorized=true`
- `paired_closed_loop_evaluation_performed=false`
- `evaluation_episodes_executed=0`
- `gate_passed=true`
- `outcome_status=SAGE_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_READY`

Lecture : le blocage de couverture cible de SAGE.8f est leve sans analogie
inter-jeux. `tn36` progresse avec le masque 858 puis ACTION6(34,51) ; `wa30`
progresse a la cinquantieme action avec ACTION5. Pour les deux jeux, une seconde
instance reproduit exactement l'etat initial, l'etat avant l'action positive,
l'etat suivant et le passage de `levels_completed=0` a `1`. Ces deux exemples
sont des signaux objectifs propres au domaine cible, pas des confirmations A33.

Suite requise : SAGE.8h doit compiler les deux trajectoires positives en une
politique de progression strictement scopee aux etats exacts de leur jeu, puis
reprendre une evaluation fermee appariee avec/sans memoire relationnelle. Les
outcomes SAGE.8h resteront reserves a l'evaluation ; ils ne devront ni choisir
les trajectoires d'apprentissage, ni regler le planner.

## SAGE.8h - exact goal-grounded memory closed-loop evaluation

Objectif :

- Compiler chaque etat pre-action des deux trajectoires positives SAGE.8g en
  une memoire `game_id|visual_digest -> action`.
- Rejouer les sources pendant la compilation et verifier les digests RESET et
  finaux sans utiliser les outcomes de SAGE.8h.
- Interdire le matching flou, le transfert inter-jeux et toute action devenue
  illegale ; revenir au planner state-conditioned SAGE.8d hors scope.
- Evaluer un episode depuis RESET par jeu, avec le meme reset, le meme horizon
  et le meme fallback dans les deux bras.
- Desactiver completement la memoire dans le temoin et l'activer avant le
  fallback dans le traitement.
- Arreter un bras seulement apres une hausse observee de `levels_completed` ou
  un etat terminal ; ces outcomes ne participent jamais au classement d'action.
- Mesurer `levels_completed` et `win_rate` comme metriques ARC principales.
- Etiqueter explicitement ce protocole comme conversion sur les trajectoires
  d'apprentissage exactes, et non comme generalisation held-out.

Ajouts :

- `theory/sage/goal_grounded_relational_memory_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_goal_grounded_relational_memory_evaluation.py`
- `diagnostics/sage/sage8h_goal_grounded_relational_memory_evaluation.json`

Run du 2026-07-19 :

- `paired_rollouts_evaluated=2`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `compiled_demonstration_transitions=57`
- `compiled_exact_visual_states=57`
- `ambiguous_exact_states=0`
- `no_memory_steps_executed=57`
- `with_memory_steps_executed=57`
- `no_memory_applications=0`
- `with_memory_applications=57`
- `with_memory_fallback_applications=0`
- `with_memory_exact_coverage_rate=1.0`
- `exact_source_sequences_replayed=2`
- `source_positive_final_digests_reproduced=2`
- `episodes_with_divergent_action_trajectories=2`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=2`
- `levels_completed_absolute_gain=2`
- `no_memory_wins=0`
- `with_memory_wins=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `primary_arc_progress_improved=true`
- `primary_arc_progress_regressed=false`
- `evaluation_outcomes_used_for_training_or_tuning=false`
- `future_outcomes_used_for_planning=false`
- `cross_game_transfer_performed=false`
- `held_out_generalization_evaluated=false`
- `primary_gain_is_replay_conversion_not_generalization=true`
- `gate_passed=true`
- `outcome_status=SAGE_GOAL_GROUNDED_MEMORY_EXACT_REPLAY_ARC_SCORE_GAIN_OBSERVED`

Lecture : pour la premiere fois dans SAGE.8, l'intervention memoire convertit
un signal local en progression ARC principale. Le temoin generique termine les
57 actions sans level-up ; le traitement reproduit les 57 decisions positives,
sans fallback, et gagne un niveau dans chaque jeu. Ce gain de +2 est causal dans
l'ablation exacte de la memoire sur ces resets, mais reste un resultat de
resubstitution : les memes trajectoires ont servi a compiler et a evaluer la
memoire. Il ne prouve ni transfert relationnel, ni resolution d'un niveau non vu.

Suite requise : SAGE.8i doit evaluer hors trajectoire d'apprentissage, par
exemple sur les niveaux suivants atteints apres les deux level-ups ou sur des
etats de depart held-out. La memoire exacte doit alors rester en quarantaine hors
scope, et toute generalisation structurelle doit etre comparee a ce fallback
sans utiliser les outcomes SAGE.8i pour apprendre ou regler la politique.

## SAGE.8i - exact memory held-out next-level evaluation

Objectif :

- Rejouer la trajectoire positive admise de chaque jeu uniquement comme setup
  pour atteindre son niveau suivant, puis commencer la mesure.
- Exclure le level-up du setup des metriques ARC de l'evaluation held-out.
- Evaluer les deux bras depuis le meme digest de niveau suivant, avec le meme
  horizon et le meme planner state-conditioned de fallback.
- Desactiver la memoire exacte dans le temoin et l'activer dans le traitement,
  sans elargir son scope `game_id|visual_digest` ni autoriser de fuzzy matching.
- Mettre en quarantaine chaque miss de scope et n'executer que des actions
  legales choisies par le fallback partage.
- Mesurer la progression uniquement apres l'entree dans le niveau held-out.
- Ne pas utiliser les outcomes SAGE.8i pour apprendre, regler ou classer les
  actions et ne pas presenter la memoire exacte comme une politique structurelle.

Ajouts :

- `theory/sage/goal_grounded_memory_held_out_evaluation.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_goal_grounded_memory_held_out_evaluation.py`
- `diagnostics/sage/sage8i_goal_grounded_memory_held_out_evaluation.json`

Run du 2026-07-19 :

- `paired_held_out_rollouts_evaluated=2`
- `held_out_levels_evaluated=2`
- `games_evaluated=[tn36-ab4f63cc, wa30-ee6fef47]`
- `training_memory_entries=57`
- `setup_actions_executed_total=114`
- `held_out_scope_keys_observed=57`
- `held_out_scope_keys_matching_training_memory=0`
- `no_memory_steps_executed=57`
- `with_memory_steps_executed=57`
- `no_memory_applications=0`
- `with_memory_applications=0`
- `with_memory_scope_misses=57`
- `with_memory_fallback_applications=57`
- `with_memory_exact_coverage_rate=0.0`
- `episodes_with_identical_action_trajectories=2`
- `episodes_with_identical_final_states=2`
- `no_memory_levels_completed_delta_total=0`
- `with_memory_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `no_memory_win_rate=0.0`
- `with_memory_win_rate=0.0`
- `win_rate_absolute_gain=0.0`
- `held_out_generalization_evaluated=true`
- `held_out_generalization_observed=false`
- `exact_memory_quarantined_out_of_scope=true`
- `structural_generalization_policy_applied=false`
- `evaluation_outcomes_used_for_training_or_tuning=false`
- `future_outcomes_used_for_planning=false`
- `cross_game_transfer_performed=false`
- `gate_passed=true`
- `outcome_status=SAGE_EXACT_GOAL_MEMORY_HELD_OUT_NEXT_LEVEL_SCOPE_SAFE_NO_GENERALIZATION`

Lecture : SAGE.8i etablit un vrai holdout par rapport aux 57 etats ayant servi
a compiler la memoire. Apres le setup, les deux niveaux suivants produisent 57
cles exactes nouvelles. La memoire n'en reconnait aucune, reste entierement en
quarantaine et laisse le meme fallback produire des trajectoires et des etats
finaux identiques dans les deux bras. Cette absence de gain n'est pas une
regression : elle confirme que le gain SAGE.8h etait une conversion de replay
scopee et que la memoire exacte ne fuit pas hors distribution. Elle confirme
aussi que SAGE ne generalise pas encore vers ces niveaux non vus.

Suite requise : SAGE.8j doit figer le mecanisme d'apprentissage avant l'examen,
mais laisser ses croyances, sa memoire causale et sa politique evoluer apres
chaque consequence observee. Les niveaux SAGE.8i n'etant plus vierges, cette
evaluation doit employer de nouveaux jeux ou etats sans charger leurs traces.
Le critere de succes reste une hausse de `levels_completed` ou de `win_rate` ;
une consequence deja observee peut guider l'action suivante, contrairement au
resultat futur d'une action qui n'a pas encore ete executee.

## SAGE.8j - online causal learning during a fresh ARC examination

Objectif :

- Selectionner deux nouveaux jeux click depuis leurs metadonnees uniquement et
  exclure explicitement les jeux et traces SAGE.8i.
- Figer avant la premiere action le code d'apprentissage, les budgets, les
  primitives structurelles et les criteres de mesure, mais pas l'etat interne
  de la politique.
- Extraire en ligne les composantes connexes, leurs formes, couleurs, positions
  et relations spatiales, tout en conservant un digest visuel exact pour audit.
- Formuler chaque sequence depuis RESET comme une hypothese action-effet et
  compter toutes les actions de replay dans le budget d'examen.
- Apres chaque effet observe, mettre a jour les croyances causales par famille
  d'action, creer les nouveaux noeuds d'etat et eliminer les doublons.
- Autoriser les outcomes deja observes a classer l'experience suivante, sans
  jamais consulter l'outcome d'une action non executee.
- Comparer cette exploration adaptative a une enumeration lexicographique dont
  l'ordre complet est fixe avant les outcomes, avec les memes bornes.
- Conserver sans retouche le premier resultat des jeux vierges ; ne verifier une
  sequence apprise sur une nouvelle instance qu'apres sa decouverte.

Ajouts :

- `theory/sage/online_causal_exam_learning.py`
- export dans `theory/sage/__init__.py`
- `tests/test_sage_online_causal_exam_learning.py`
- `diagnostics/sage/sage8j_online_causal_exam_learning.json`

Run du 2026-07-19 :

- `fresh_games_evaluated=2`
- `games_evaluated=[lf52-271a04aa, lp85-305b61c3]`
- `max_action_executions_per_game_per_arm=512`
- `max_trials_per_game_per_arm=256`
- `max_sequence_depth=4`
- `control_action_executions=608`
- `adaptive_action_executions=514`
- `control_trials_executed=233`
- `adaptive_trials_executed=166`
- `adaptive_belief_updates=166`
- `adaptive_hypothesis_revisions=6`
- `adaptive_discovered_structural_states=26`
- `adaptive_duplicate_states_pruned=71`
- `adaptive_no_effect_states_pruned=0`
- `control_levels_completed_delta_total=0`
- `adaptive_levels_completed_delta_total=0`
- `levels_completed_absolute_gain=0`
- `control_wins=0`
- `adaptive_wins=0`
- `adaptive_answers_learned_during_exam=0`
- `online_learning_during_exam_performed=true`
- `learning_algorithm_frozen_before_exam=true`
- `agent_policy_state_frozen_during_exam=false`
- `observed_outcomes_used_for_next_action_selection=true`
- `future_outcomes_used_for_action_selection=false`
- `sage8i_action_traces_loaded=false`
- `game_source_files_inspected_by_agent=[]`
- `primary_arc_progress_improved=false`
- `primary_arc_progress_regressed=false`
- `gate_passed=true`
- `outcome_status=SAGE_ONLINE_CAUSAL_EXAM_LEARNING_ACTIVE_NO_ARC_GAIN`

Lecture : SAGE.8j apprend desormais pendant l'examen au sens demande par ARC.
Son algorithme reste fixe, mais ses hypotheses et son graphe causal changent
apres chaque action. Sur les deux premiers jeux vierges, il a caracterise six
familles d'action, revise six hypotheses, construit 26 etats structurels et
evite 71 explorations redondantes. Il n'a cependant trouve ni level-up ni WIN
dans les bornes fixees. Ce jalon valide donc la boucle d'apprentissage en ligne,
pas encore son efficacite pour resoudre une tache ARC inconnue.

Suite requise : SAGE.8k doit exploiter ce diagnostic pour apprendre des effets
orientes objet et objectif, puis planifier des experiences discriminantes plutot
que seulement explorer le graphe en profondeur. Cette nouvelle regle devra etre
figee avant de nouveaux jeux vierges ; `lf52` et `lp85` deviennent des jeux de
developp et ne pourront plus servir de preuve held-out.

## SAGE.8k - online terminal-objective grounding in the unified controller

Objectif :

- Separer strictement une regle mecanique confirmee d'une hypothese de but.
- Deriver seulement des objectifs directionnels et mesurables : transformer
  une couleur source vers une cible, ou etablir une relation actuellement
  absente.
- Ne pas transformer une relation simplement preservee en objectif terminal.
- Mesurer une distance avant/apres chaque option sans compter le changement
  visuel ou le succes mecanique comme preuve terminale.
- Autoriser un petit budget de sondes par objectif candidat, puis arreter les
  repetitions quand le deficit est nul ou le budget epuise.
- Crediter une reduction seulement si un `level_complete` ou un WIN est observe
  dans une fenetre causale courte ; refuter les completions repetees suivies de
  GAME_OVER.
- Exploiter sans budget de sonde uniquement les objectifs ayant recu ce support
  terminal observe.
- Exposer objectif, statut et distance dans chaque decision et dans le benchmark
  apparie held-out.

Ajouts :

- `theory/online_terminal_objective.py`
- branchement dans `theory/unified_cognitive_controller.py`
- metriques v2 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_terminal_objectives.py`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Run du 2026-07-19, memes 5 jeux public-unseen, seeds 0/1, 2 resets,
40 actions par reset :

- `paired_protocol.protocol_gate_passed=true`
- `legacy_only.actions_executed=768`
- `unified.actions_executed=798`
- `unified.controller_errors=0`
- `unified.experiment_actions=294`
- `unified.promoted_relational_rules_observed=19` sur les 10 runs
- `unified.terminal_objective_probe_actions=0`
- `unified.terminal_objective_grounded_actions=0`
- `unified.promoted_option_actions=0`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`
- `targeted_grounding_tests=20 passed`
- `full_repository_tests=1327 passed`
- `ruff_and_compileall=passed`

Lecture : le nouveau grounding retire les 34 actions d'options du run precedent.
Les 19 regles promues dans ce benchmark sont des relations `absent` ou
`preserved`; aucune ne definit un deficit terminal directionnel. Elles restent
des connaissances mecaniques mais ne sont plus poursuivies comme des buts.
Les tests synthetiques prouvent separement les quatre comportements requis : un
effet mecanique seul reste candidat, une reduction suivie d'un level-up devient
`terminal_supported`, un level-up hors fenetre ne donne aucun credit, et deux
completions suivies de GAME_OVER refutent l'objectif. Il n'y a encore aucun gain
ARC held-out.

Suite requise : le prochain verrou n'est plus l'execution des options mais la
generation active d'hypotheses de but directionnelles lorsque les premieres
regles apprises ne decrivent que `absent`/`preserved`. Il faudra proposer des
interventions qui peuvent faire apparaitre, casser, epuiser, atteindre ou
convertir une structure, puis laisser exclusivement le signal terminal en ligne
departager ces objectifs concurrents sur de nouveaux episodes held-out.

## SAGE.8l - active online goal generation and terminal discrimination

Objectif :

- Generer en ligne cinq familles de buts mesurables : `appear`, `break`,
  `exhaust`, `reach` et `convert`.
- Deriver les candidats uniquement de la structure live, des actions legales et
  des mecaniques apprises ; les priorites de generation ne comptent jamais comme
  support de but.
- Borner globalement la banque a 10 candidats et a 2 candidats par famille,
  avec eviction seulement des candidats plus faibles encore intacts.
- Choisir des interventions qui devraient reduire un sous-ensemble des distances
  concurrentes, et exposer les buts affectes/non affectes dans chaque decision.
- Utiliser la meme action pour reviser simultanement les hypotheses mecaniques,
  sans convertir ce succes mecanique en preuve terminale.
- Mesurer toutes les reductions reelles avant l'evenement terminal, mais limiter
  la transition de changement d'ecran `level_complete` aux reductions annoncees
  avant l'action.
- Mettre en quarantaine tout level-up precede de plusieurs buts reduits comme
  `ambiguous_terminal`, sans confirmation automatique.
- Exiger deux contextes terminaux independants ou un contraste terminal par
  intervention alternative avant `terminal_supported`.
- Programmer une ablation apres le premier support provisoire ; un terminal sans
  reduction du but conteste sa necessite sans refuter sa mecanique.
- Imputer GAME_OVER a l'intervention dangereuse, pas au but, et exclure ensuite
  cette intervention pour ce but.
- Compiler aussi les mecaniques relationnelles `broken` en options dirigees.

Ajouts :

- `theory/online_goal_hypothesis.py`
- extension de `theory/online_terminal_objective.py`
- integration dans `theory/unified_cognitive_controller.py`
- extension `broken` dans `theory/promoted_relational_rule.py` et
  `theory/online_relational_option.py`
- metriques v3 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_goal_hypothesis.py`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Run du 2026-07-19, memes 5 jeux public-unseen, seeds 0/1, 2 resets,
40 actions par reset :

- `paired_protocol.protocol_gate_passed=true`
- `legacy_only.actions_executed=768`
- `unified.actions_executed=800`
- `unified.controller_errors=0`
- `unified.experiment_actions=356`
- `unified.experiment_cost_rate=0.445`
- `unified.generated_goal_hypotheses=76` sommes des banques finales des 10 runs
- `unified.terminal_objective_discriminator_actions=88`
- `unified.terminal_objective_probe_actions=12`
- `unified.terminal_objective_ablation_actions=0`
- `unified.terminal_objective_grounded_actions=0`
- `unified.objective_distance_reductions=667`
- `unified.objective_nonterminal_completions=10`
- `unified.refuted_objectives=2`
- `unified.objective_ambiguous_terminal_events=0`
- `unified.terminal_supported_objectives=0`
- `unified.unsafe_goal_plan_failures=0`
- `online_mechanic_hypothesis_revisions=332`
- `promoted_relational_rules_observed=34` sommes des 10 runs
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`
- `targeted_tests=30 passed` apres le dernier test d'ablation
- `full_repository_tests=1338 passed`
- `ruff_and_compileall=passed`

Lecture : SAGE ne reste plus bloque lorsque les premieres regles promues sont
`absent` ou `preserved`. Il construit une banque diversifiee, mesure les cinq
types de deficit et consomme 100 actions explicitement orientees vers la
discrimination de buts. Dix completions locales sans signal terminal produisent
de l'evidence negative et deux objectifs sont refutes. Aucun level-up ni WIN
n'est cependant observe ; il n'existe donc honnetement ni support terminal, ni
ablation declenchee, ni objectif exploitable dans ce run. Les 667 reductions de
distance restent des observations locales, pas un gain ARC.

Suite requise : le prochain verrou est la composition temporelle. Les hypotheses
de but sont maintenant generees et testees, mais les interventions restent
principalement primitives. Il faut apprendre des sous-objectifs ordonnes et des
sequences state-conditioned qui maintiennent une direction de deficit sur un
horizon plus long, tout en conservant le meme arbitre terminal et sans regler la
politique sur les outcomes de ce benchmark.

## SAGE.8m - online state-conditioned temporal goal composition

Objectif :

- Composer des plans de sous-objectifs plutot que des listes d'actions figees.
- Decomposer tout deficit superieur a un en paliers de distance ordonnes, puis
  reevaluer la garde d'etat apres chaque intervention primitive.
- Construire aussi des dependances entre hypotheses live, par exemple convertir
  une structure pour faire apparaitre une couleur absente avant de poursuivre
  `appear` ou `reach`.
- Cibler explicitement le sous-objectif courant dans le designer d'experiences,
  sans perdre la revision simultanee des hypotheses mecaniques.
- N'executer qu'une intervention a la fois ; continuer seulement si la distance
  mesuree baisse, avancer seulement si le seuil courant est atteint.
- Abandonner et autoriser une autre chaine apres stagnation, garde devenue
  non mesurable, budget epuise, intervention indisponible ou veto de securite.
- Imputer GAME_OVER a la chaine et a son intervention, sans refuter le but
  terminal sous-jacent.
- Conserver les completions locales comme evidence de controle uniquement.
  Une chaine ne recoit du support qu'apres un vrai level-up/WIN dans sa fenetre
  causale, et exige deux contextes terminaux independants avant exploitation.
- Exposer dans chaque decision l'identite de la chaine, son statut terminal,
  l'index du sous-objectif, le nombre d'etapes, la distance courante et le seuil.

Ajouts :

- `theory/online_temporal_goal_composition.py`
- extension ciblee de `theory/online_goal_hypothesis.py`
- integration dans `theory/unified_cognitive_controller.py`
- metriques v4 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_temporal_goal_composition.py`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Run du 2026-07-19, memes 5 jeux public-unseen, seeds 0/1, 2 resets,
40 actions par reset :

- `paired_protocol.protocol_gate_passed=true`
- `legacy_only.actions_executed=768`
- `unified.actions_executed=800`
- `unified.controller_errors=0`
- `unified.experiment_actions=464`
- `unified.experiment_cost_rate=0.58`
- `unified.generated_goal_hypotheses=76`
- `unified.temporal_plans_generated=132`
- `unified.temporal_plan_starts=120`
- `unified.temporal_subgoal_probe_actions=188`
- `unified.temporal_progress_events=40`
- `unified.temporal_step_completions=12`
- `unified.temporal_local_completions=2`
- `unified.temporal_plan_stalls=148`
- `unified.temporal_plan_abandonments=118`
- `unified.temporal_unsafe_failures=0`
- `unified.terminal_supported_temporal_plans=0`
- `unified.objective_distance_reductions=646`
- `unified.terminal_supported_objectives=0`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`
- `new_temporal_composition_tests=8 passed`
- `full_repository_tests=1346 passed`
- `ruff_and_py_compile=passed`

Lecture : le verrou logiciel de composition temporelle est leve. SAGE a
effectivement transforme les buts primitifs en 132 hypotheses de chaines, a
execute 188 interventions par sous-objectif et a reevalue l'etat entre chacune.
Quarante transitions ont maintenu la direction du deficit, douze paliers ont ete
franchis et deux chaines ont ete completees localement. Elles ne sont pas
promues : aucun level-up ni WIN n'a fourni le credit terminal requis.

Le diagnostic est egalement net : 148 stagnations conduisent a 118 abandons et
le cout experimental monte a 58 %. Le prochain verrou n'est donc plus
l'ordonnancement temporel lui-meme, mais l'induction de preconditions causales
et de sous-objectifs intermediaires réellement habilitants. Il faudra apprendre
depuis les transitions quelles transformations rendent une intervention future
possible, construire ce graphe de dependances au-dela du seul cas de couleur
manquante, puis arbitrer les chaines par probabilite de progres observee et cout
avant de les deployer sur de nouveaux episodes held-out.

## SAGE.8n - online causal subgoal induction and utility arbitration

Objectif :

- Detecter les objectifs temporairement bloques lorsqu'aucune intervention sure
  ne peut reduire leur deficit dans l'etat courant.
- Proposer en ligne un petit nombre de preconditions candidates a partir des
  objectifs mesurables qui partagent une structure, avec une sonde generique
  bornee quand aucun recouvrement n'est disponible.
- Apprendre une arete `source -> cible` seulement si une reduction reelle de la
  source rend ensuite la cible testable, mesurable ou moins distante.
- Refuser explicitement comme preuve causale un changement de la cible sans
  reduction concomitante de la source.
- Exiger deux contextes independants avant de confirmer une arete mecanique et
  deux echecs independants avant de la refuter.
- Garder le statut causal mecanique strictement separe du support terminal du
  but et du plan.
- Transferer les preconditions entre dispositions spatiales equivalentes avec
  une signature d'etat invariante aux positions exactes.
- Compiler les aretes pertinentes en plans temporels `precondition -> cible`,
  toujours executes une action puis une observation a la fois.
- Arbitrer tous les plans avec une utilite apprise combinant probabilite de
  progres, probabilite de franchissement, cout observe, abandons et risque.
- Exposer arete causale, probabilite, cout et utilite dans chaque decision.
- Fournir une ablation reproductible qui desactive seulement SAGE.8n.

Ajouts :

- `theory/online_causal_subgoal_graph.py`
- extension de `theory/online_temporal_goal_composition.py`
- integration dans `theory/unified_cognitive_controller.py`
- metriques et ablation v5 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_causal_subgoal_graph.py`
- mise a jour de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8n_cn04_utility_audit.json`
- `diagnostics/sage/sage8n_cn04_causal_ablation.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `paired_protocol.protocol_gate_passed=true`
- `paired_protocol.causal_subgoal_induction_enabled_in_unified=true`
- `legacy_only.actions_executed=768`
- `unified.actions_executed=800`
- `unified.controller_errors=0`
- `unified.experiment_actions=474`
- `unified.experiment_cost_rate=0.5925`
- `unified.generated_goal_hypotheses=76`
- `unified.causal_edges_generated=88`
- `unified.causal_blocked_target_events=26`
- `unified.causal_edge_trials=44`
- `unified.causal_edge_actions=78`
- `unified.causal_edge_source_progress_events=4`
- `unified.causal_edge_support_events=2`
- `unified.causal_availability_successes=2`
- `unified.causal_cochange_supports=0`
- `unified.causal_edge_plan_failures=28`
- `unified.confirmed_causal_edges=0`
- `unified.refuted_causal_edges=0`
- `unified.causal_dependency_plans=62`
- `unified.causal_dependency_plan_starts=68`
- `unified.causal_dependency_plan_actions=78`
- `unified.causal_dependency_progress_events=4`
- `unified.causal_dependency_step_completions=2`
- `unified.temporal_plans_generated=170`
- `unified.temporal_plan_actions=190`
- `unified.temporal_progress_events=30`
- `unified.temporal_step_completions=14`
- `unified.temporal_plan_stalls=160`
- `unified.temporal_plan_abandonments=118`
- `unified.temporal_local_completions=2`
- `unified.temporal_nonterminal_completions=2`
- `unified.objective_distance_reductions=740`
- `unified.objective_nonterminal_completions=83`
- `unified.refuted_objectives=11`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`
- `new_sage8n_tests=10 passed`
- `targeted_cognitive_tests=40 passed`
- `full_repository_tests=1356 passed`
- `ruff_and_compileall=passed`

Audit cible `cn04-65d47d14`, seed 1, memes 2 resets x 40 :

- SAGE.8n active : `levels_completed=0`, `causal_edge_actions=15`,
  `causal_availability_successes=1`, `causal_edge_support_events=1`,
  `experiment_cost_rate=0.675`.
- Ablation SAGE.8n : `levels_completed=0`, `causal_edge_actions=0`,
  `experiment_cost_rate=0.7625`.
- L'induction causale remplace une partie de l'exploration generique par des
  essais structures et reduit ici le cout experimental, sans franchir le niveau.

Lecture : SAGE.8n apprend et execute bien des preconditions causales en ligne,
mais ne produit pas encore de gain ARC final. Une version intermediaire a utilite
causale figee et optimiste avait atteint un niveau sur `cn04`; le resultat
disparait des que l'utilite est correctement remise a jour avec les couts,
abandons et risques observes. Cette sensibilite ne constitue donc pas un gain
reproductible et n'est pas retenue. La version finale conserve le classement
dynamique : elle observe deux recuperations de disponibilite, mais aucune arete
n'obtient les deux contextes independants requis pour etre confirmee et aucun
objectif ni plan n'est `terminal_supported`.

Le prochain verrou est de rendre la preparation causale assez productive pour
etre consolidee : seulement 4 progres de source sur 78 actions causales, avec
28 echecs de plan. Il faut mieux representer les effets qui ouvrent reellement
une intervention, attribuer le credit sur plusieurs etapes et resets, puis
obtenir des confirmations independantes transferables sans relacher la preuve
terminale. C'est seulement ensuite que ces dependances pourront devenir des
options hierarchiques capables de gagner davantage de niveaux.

## SAGE.8o - effect-conditioned causal credit and cross-reset confirmation

Objectif :

- Abstraire chaque effet causal observe sans memoriser les positions absolues :
  variations de couleurs, volume de changement, mouvement, redimensionnement et
  signaux terminaux restent separes.
- Memoriser, par arete causale, quelles interventions semantiques produisent un
  progres de source puis rendent effectivement la cible testable.
- Transferer les clics par couleur et les actions non parametrees par famille
  d'intervention plutot que par coordonnees exactes.
- Conserver un essai causal apres une progression partielle et lui attribuer le
  credit lorsque la disponibilite de la cible est constatee plus tard.
- Expirer ce credit apres une fenetre bornee sans transformer une censure de
  budget en contradiction.
- Exiger des branches distinctes, et non seulement des signatures d'etat
  distinctes dans un meme reset, pour confirmer ou refuter une arete.
- Donner a une arete soutenue une priorite de confirmation au reset suivant.
- Reserver au plus deux demarrages de confirmation par branche lorsque le
  budget normal des plans temporels est deja epuise.
- Guider le nouvel essai avec l'intervention productive apprise et continuer a
  executer une seule action avant re-observation.
- Garder la confirmation mecanique strictement separee de la preuve que le but
  ou le plan a une valeur terminale.
- Fournir une ablation qui conserve SAGE.8n mais desactive uniquement la memoire
  d'effets, le credit retarde et les confirmations reservees de SAGE.8o.

Ajouts :

- extension de `theory/online_causal_subgoal_graph.py`
- extension de `theory/online_goal_hypothesis.py`
- extension de `theory/online_temporal_goal_composition.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v6 dans `theory/unified_cognition_ab_benchmark.py`
- extension de `tests/test_online_causal_subgoal_graph.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8o_cn04_causal_confirmation.json`
- `diagnostics/sage/sage8o_effect_credit_ablation.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v6`
- `paired_protocol.protocol_gate_passed=true`
- `paired_protocol.causal_subgoal_induction_enabled_in_unified=true`
- `paired_protocol.causal_effect_credit_enabled_in_unified=true`
- `legacy_only.actions_executed=768`
- `unified.actions_executed=800`
- `unified.controller_errors=0`
- `unified.experiment_actions=476`
- `unified.experiment_cost_rate=0.595`
- `unified.causal_edge_actions=83`
- `unified.causal_edge_source_progress_events=6`
- `unified.causal_edge_support_events=4`
- `unified.causal_availability_successes=4`
- `unified.causal_effect_observations=83`
- `unified.causal_effect_guided_actions=5`
- `unified.causal_productive_effect_signatures=2`
- `unified.causal_productive_intervention_signatures=2`
- `unified.causal_delayed_credit_events=4`
- `unified.causal_reserved_confirmation_starts=3`
- `unified.causal_cross_branch_confirmations=2`
- `unified.confirmed_causal_edges=2`
- `unified.refuted_causal_edges=0`
- `unified.objective_distance_reductions=730`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `causal_effect_credit_enabled_in_unified=false`
- `causal_edge_actions=78`
- `causal_edge_source_progress_events=4`
- `causal_edge_support_events=2`
- `causal_effect_observations=0`
- `causal_effect_guided_actions=0`
- `causal_reserved_confirmation_starts=0`
- `causal_cross_branch_confirmations=0`
- `confirmed_causal_edges=0`
- `levels_completed=0`

Audit cible `cn04-65d47d14`, seed 1, 2 resets x 40 :

- l'arete `reach(color8) -> appear(same_shape(colors4,8))` recoit deux supports
  sur deux branches, zero contradiction et devient `confirmed` ;
- `ACTION1` est apprise comme intervention productive transferable avec trois
  progres sources et deux ouvertures de la cible ;
- `causal_effect_guided_actions=3` ;
- `causal_reserved_confirmation_starts=2` ;
- `confirmed_causal_edges=1` contre zero avant SAGE.8o ;
- `levels_completed=0` et aucun objectif ou plan n'est `terminal_supported`.

Validation :

- `new_sage8o_tests=7 passed`
- `targeted_cognitive_tests=41 passed`
- `full_repository_tests=1363 passed`
- `ruff_and_compileall=passed`

Lecture : le verrou de consolidation causale multi-reset est franchi. SAGE ne
se contente plus de proposer des dependances : il reconnait un effet productif,
le rejoue dans une branche independante et confirme la meme relation mecanique.
L'ablation reproduit exactement l'ancien plafond de SAGE.8n, avec deux supports
isoles et aucune arete confirmee. Le cout experimental augmente legerement de
`0.5925` a `0.595`, tandis que les reductions d'objectif passent de 740 a 730 :
il ne faut donc pas presenter cette etape comme un gain d'efficacite ou de
niveau. Elle transforme une intuition causale fragile en connaissance
transferable et falsifiable.

Le prochain verrou est l'exploitation terminale des dependances confirmees.
Apres avoir ouvert une intervention, SAGE doit tester activement les buts aval,
attribuer un eventuel level-up a la chaine complete, ou refuter rapidement les
buts aval non terminaux. Les aretes confirmees doivent alors etre compilees en
options hierarchiques reutilisables qui liberent du budget pour chercher la
derniere transformation terminale plutot que rejouer indefiniment la
preparation.

## SAGE.8p - confirmed causal options and terminal suffix search

Objectif :

- Compiler uniquement les aretes causales confirmees en options hierarchiques.
- Activer une option seulement apres l'observation effective de l'ouverture de
  sa cible, sans confondre confirmation mecanique et valeur terminale.
- Explorer alors les transformations aval avec un budget borne a six actions
  et trois essais par signature semantique.
- Preferer les mecanismes promus par l'apprentissage en ligne, la nouveaute des
  familles d'action et les interventions dont l'utilite observee est positive.
- Memoriser les effets aval, progres objectifs, completions de cible, risques et
  sequences terminales sur toute la chaine causale.
- Attribuer un succes terminal a l'option seulement lorsque l'action terminale
  participe a son rollout actif ; ignorer les terminaux sans rapport.
- Exiger deux contextes independants avant de promouvoir une sequence terminale
  et deux branches contradictoires avant de refuter la valeur terminale d'une
  option.
- Traiter les resets et fins de budget comme des censures, pas comme des preuves
  negatives.
- Rejouer en priorite un suffixe terminal appris, tout en conservant la preuve
  terminale du but separee de celle de l'option.
- Fournir une ablation qui conserve SAGE.8o et desactive seulement les options
  causales hierarchiques de SAGE.8p.

Ajouts :

- `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v7 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_causal_option.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8p_causal_option_ablation.json`
- `diagnostics/sage/sage8p_cn04_downstream_search.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v7`
- `paired_protocol.protocol_gate_passed=true`
- `paired_protocol.causal_hierarchical_options_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.confirmed_causal_edges=2`
- `unified.causal_options_compiled=2`
- `unified.causal_option_opening_events=2`
- `unified.causal_option_rollouts=2`
- `unified.causal_option_downstream_actions=12`
- `unified.causal_option_downstream_effects=8`
- `unified.causal_option_downstream_progress_events=0`
- `unified.causal_option_target_completions=0`
- `unified.causal_option_nonterminal_rollouts=2`
- `unified.causal_option_unsafe_rollouts=0`
- `unified.causal_option_terminal_credited_events=0`
- `unified.terminal_supported_causal_options=0`
- `unified.terminal_refuted_causal_options=0`
- `unified.causal_option_censored_openings=0`
- `unified.objective_distance_reductions=740`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `causal_hierarchical_options_enabled_in_unified=false`
- `experiment_actions=476`
- `experiment_cost_rate=0.595`
- `confirmed_causal_edges=2`
- `causal_options_compiled=0`
- `causal_option_downstream_actions=0`
- `causal_option_downstream_effects=0`
- `objective_distance_reductions=730`
- `levels_completed=0`

Audit cible `cn04-65d47d14`, seed 1, 2 resets x 40 :

- une option est compilee puis ouverte a partir de l'arete confirmee
  `reach(color8) -> appear(same_shape(colors4,8))` ;
- elle prend le controle de la recherche aval pendant six actions bornees ;
- quatre de ces actions produisent un effet observable ;
- les suffixes aval testes couvrent notamment `ACTION4`, `ACTION6::color8` et
  `ACTION2` plutot que de rejouer seulement la preparation ;
- aucun progres aval, aucune completion de cible et aucun terminal ne sont
  observes ;
- `levels_completed=0`.

Validation :

- `new_sage8p_tests=10 passed`
- `targeted_cognitive_tests=34 passed`
- `full_repository_tests=1373 passed`
- `ruff_and_compileall=passed`

Lecture : le verrou de compilation et d'exploitation des dependances causales
confirmees est franchi. Une ouverture confirmee devient maintenant un contexte
d'exploration aval explicite, borne et auditable. Huit actions aval sur douze
produisent un effet, et les reductions de distance objective passent de 730 a
740 face a l'ablation. Cette recherche consomme toutefois douze actions
experimentales supplementaires (`0.595` a `0.61`) et ne produit encore ni
progres aval mesure, ni preuve terminale, ni niveau gagne. Il ne faut donc pas
la presenter comme un gain ARC reproductible.

Le prochain verrou est de transformer les effets aval nouvellement observes en
hypotheses de sous-buts multi-etapes : relier chaque effet a une transformation
objective mesurable, enchainer les suffixes au-dela d'une seule famille
d'action et concentrer le budget sur les branches qui reduisent effectivement
la distance au but. La preuve finale doit rester exclusivement terminale et
apprise pendant l'examen.

## SAGE.8q - effect-conditioned downstream subgoals

Objectif :

- Generer de nouveaux buts mesurables a partir des transitions aval reellement
  observees : depletion ou flux de couleurs, bascule de relation et approche
  d'une structure.
- Filtrer les couleurs de fond dominantes afin de ne pas transformer un simple
  changement d'affichage global en pseudo-objet sans controle.
- Relier chaque effet abstrait, dans le contexte d'une option confirmee, aux
  objectifs qu'il rend mesurables ou dont il reduit la distance.
- Garder ces liens strictement mecaniques : une reduction de distance du
  declencheur ne donne aucune preuve terminale et ne prouve pas encore qu'un
  suffixe aval sait poursuivre le but.
- Distinguer le progres du declencheur du progres d'une poursuite aval ; seule
  une reduction obtenue apres la selection du sous-but rend celui-ci
  `progress_supported`.
- Ne jamais rejouer comme suffixe l'action qui a seulement revele le contexte ;
  memoriser et rejouer uniquement une sequence aval ayant produit un progres
  ulterieur reel.
- Selectionner les interventions par objectif avec le concepteur d'experiences
  existant, puis re-observer apres chaque primitive.
- Limiter une branche a deux actions par sous-but et par objectif afin de
  diversifier les transformations testees au lieu de repeter une piste sterile.
- Utiliser un budget aval adaptatif : quatre actions de base, puis deux actions
  supplementaires par transition productive, dans la limite SAGE.8p de six.
- Refuter une poursuite seulement apres deux branches independantes sans
  progres ; traiter les changements de piste, resets, terminaux et risques
  comme des censures.
- Conserver l'attribution terminale de l'option et des buts exclusivement liee
  aux level-up et WIN observes pendant l'examen.
- Fournir une ablation qui conserve SAGE.8p et desactive seulement la generation
  de buts par effet, leur poursuite et le budget adaptatif de SAGE.8q.

Ajouts :

- `theory/online_effect_conditioned_subgoal.py`
- extension de `theory/online_goal_hypothesis.py`
- extension de `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v8 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_effect_conditioned_subgoal.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8q_effect_subgoal_ablation.json`
- `diagnostics/sage/sage8q_cn04_effect_subgoal_search.json`
- `diagnostics/sage/sage8q_cn04_effect_subgoal_ablation.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v8`
- `paired_protocol.protocol_gate_passed=true`
- `effect_conditioned_downstream_subgoals_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.experiment_actions=486`
- `unified.experiment_cost_rate=0.6075`
- `unified.confirmed_causal_edges=2`
- `unified.causal_options_compiled=2`
- `unified.causal_option_downstream_actions=12`
- `unified.causal_option_downstream_effects=4`
- `unified.effect_conditioned_goal_candidates_generated=6`
- `unified.effect_conditioned_subgoals_generated=16`
- `unified.effect_conditioned_subgoal_links=16`
- `unified.productive_effect_subgoal_links=4`
- `unified.effect_conditioned_subgoal_selections=10`
- `unified.effect_conditioned_subgoal_guided_actions=10`
- `unified.effect_conditioned_subgoal_progress_events=4`
- `unified.effect_conditioned_trigger_progress_events=4`
- `unified.effect_conditioned_pursuit_progress_events=0`
- `unified.progress_supported_effect_conditioned_subgoals=0`
- `unified.failed_effect_conditioned_subgoal_pursuits=2`
- `unified.censored_effect_conditioned_subgoal_pursuits=8`
- `unified.causal_option_dynamic_budget_extensions=2`
- `unified.objective_distance_reductions=732`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `effect_conditioned_downstream_subgoals_enabled_in_unified=false`
- `experiment_actions=488`
- `experiment_cost_rate=0.61`
- `confirmed_causal_edges=2`
- `causal_options_compiled=2`
- `causal_option_downstream_actions=12`
- `causal_option_downstream_effects=8`
- toutes les metriques de sous-buts conditionnes par effet valent zero ;
- `objective_distance_reductions=740`
- `levels_completed=0`

Audit cible `cn04-65d47d14`, seed 1, 2 resets x 40 :

- 3 candidats de but sont generes depuis les effets observes ;
- 8 liens `effet -> sous-but` sont crees, dont 2 reduisent immediatement une
  distance objective ;
- 5 actions aval sont guidees et reparties entre conversion, epuisement et
  relation, avec au plus deux actions par objectif ;
- une reduction du declencheur etend le budget de quatre a six actions ;
- aucune des cinq actions de poursuite ne reduit ensuite son sous-but, donc
  aucun lien n'est `progress_supported` et aucune sequence n'est rejouee ;
- le cout experimental passe de `0.7625` a `0.75`, tandis que les reductions
  objectives passent de 27 a 23 ;
- `levels_completed=0`.

Validation :

- `new_sage8q_tests=9 passed`
- `targeted_cognitive_tests=28 passed`
- `full_repository_tests=1382 passed`
- `ruff_and_compileall=passed`

Lecture : SAGE.8q franchit le verrou representationnel. Un effet aval n'est plus
seulement une signature de changement : il peut engendrer un but mesurable,
ouvrir une poursuite multi-etapes, etre compare a d'autres buts et recevoir un
budget proportionnel a son progres observe. L'ablation confirme que ces liens,
selections et extensions proviennent uniquement de SAGE.8q. Le faible gain de
cout experimental (`0.61` a `0.6075`) ne compense toutefois pas la baisse des
reductions objectives (740 a 732), et aucun niveau n'est gagne. Les quatre
progres mesures sont ceux des effets declencheurs ; aucun suffixe aval ne sait
encore les prolonger.

Le prochain verrou est donc le controle directionnel dans l'etat produit par
l'effet. SAGE doit apprendre qu'une meme primitive peut inverser son effet selon
le mode courant, predire le signe de la variation objective avant l'action et
composer seulement les transitions dont la direction est compatible. Cette
representation d'etat latent et de reversibilite doit permettre d'obtenir les
premiers progres de poursuite, sans promouvoir de but avant une preuve
terminale en ligne.

## SAGE.8r - state-conditioned directional effect control

Objectif :

- Construire en ligne une signature de mode latent conditionnee par l'objectif,
  mais invariante a la position absolue des objets.
- Apprendre le signe de la variation objective pour chaque combinaison
  `option x objectif x mode x action semantique` exclusivement depuis les
  transitions observees pendant l'examen.
- Distinguer les effets progressifs, regressifs, neutres, instables et encore
  inconnus sans transformer cette evidence mecanique en preuve terminale.
- Detecter qu'une meme action peut etre reversible : progressive dans un mode
  et regressive dans un autre.
- Donner une priorite bornee a un contraste lorsqu'une action connue est
  rencontree dans un nouveau mode, puis fermer ce contraste apres un seul essai
  sans progres.
- Reutiliser en priorite une action progressive dans un mode recurrent et
  exclure une action deja regressive ou neutre dans ce mode, sauf si une
  sequence terminale ou de progres deja soutenue impose son replay.
- Crediter separement les observations du declencheur et celles de la poursuite
  afin de ne pas compter deux fois la meme transition.
- Exposer le mode, le statut directionnel, le gain attendu, la confiance et la
  reversibilite dans chaque decision auditable du controleur unifie.
- Fournir une ablation qui conserve SAGE.8q et desactive seulement le controle
  directionnel conditionne par l'etat.

Ajouts :

- `theory/online_state_conditioned_effect.py`
- extension de `theory/online_effect_conditioned_subgoal.py`
- extension de `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v9 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_state_conditioned_effect.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8r_directional_control_ablation.json`
- `diagnostics/sage/sage8r_cn04_directional_control.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v9`
- `paired_protocol.protocol_gate_passed=true`
- `state_conditioned_directional_control_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.directional_effect_observations=36`
- `unified.directional_progress_events=10`
- `unified.directional_regression_events=10`
- `unified.directional_stall_events=16`
- `unified.directional_latent_modes=16`
- `unified.directional_mode_action_models=28`
- `unified.directional_reversible_action_objectives=6`
- `unified.directional_predictions=66`
- `unified.directional_mode_contrast_selections=2`
- `unified.directional_progressive_selections=2`
- `unified.effect_conditioned_subgoal_progress_events=12`
- `unified.effect_conditioned_trigger_progress_events=10`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `state_conditioned_directional_control_enabled_in_unified=false`
- toutes les metriques directionnelles valent zero ;
- `experiment_actions=486`
- `experiment_cost_rate=0.6075`
- `effect_conditioned_subgoal_progress_events=4`
- `effect_conditioned_trigger_progress_events=4`
- `effect_conditioned_pursuit_progress_events=0`
- `progress_supported_effect_conditioned_subgoals=0`
- `objective_distance_reductions=732`
- `levels_completed=0`
- `wins=0`

Audit cible `cn04-65d47d14`, seed 0, 2 resets x 40 :

- 18 observations directionnelles couvrent 8 modes latents et 14 modeles
  `mode x action` ;
- 3 couples `action x objectif` sont detectes comme reversibles ;
- un contraste de nouveau mode est selectionne, puis une action progressive
  est reutilisee dans le mode recurrent ;
- 5 actions de poursuite produisent le premier progres aval mesure de cette
  chaine ;
- un sous-but devient `progress_supported` ;
- les reductions objectives atteignent 48 ;
- aucun niveau ni WIN n'est observe.

Validation :

- `new_sage8r_tests=5 passed`
- `targeted_cognitive_tests=34 passed`
- `full_repository_tests=1388 passed`
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou du controle directionnel local est franchi. Contrairement
a SAGE.8q, le run principal contient maintenant deux reductions obtenues apres
la selection explicite d'un sous-but, et deux sous-buts sont soutenus par cette
evidence de poursuite. L'ablation ramene exactement ces deux metriques a zero et
retire 50 reductions objectives. Ce gain coute deux actions experimentales
supplementaires sur 800 (`0.6075` a `0.61`). Aucun niveau n'est encore gagne :
ce resultat ne constitue donc pas un progres ARC terminal reproductible.

Le prochain verrou est la composition persistante des transitions
directionnelles. SAGE doit compiler les modes recurrents et leurs actions
progressives en une politique de suffixe multi-etapes, maintenir le meme but au
dela d'une reduction locale, et allouer davantage de budget aux sequences dont
le progres se repete. La promotion du but doit rester reservee a un level-up ou
WIN observe en ligne ; les reductions intermediaires servent uniquement a
chercher cette preuve terminale.

## SAGE.8s - progress-gated persistent pursuit and mode bridges

Objectif :

- Maintenir un sous-but apres sa premiere reduction objective au lieu de
  l'abandonner automatiquement a la limite fixe de deux actions de SAGE.8q.
- N'accorder cette persistance qu'apres un progres de poursuite reel ; les
  progres du declencheur, les priorites et les hypotheses seules ne debloquent
  aucun budget.
- Etendre progressivement la limite par sous-but de 2 a 4 puis 6 actions, le
  rollout de 6 a 8 puis 10 actions et la fenetre de credit de 8 a 12 puis 16
  transitions.
- Conditionner toute action supplementaire a une preuve directionnelle : action
  progressive, contraste de mode non epuise ou pont observe vers un mode ou une
  autre action est progressive.
- Compiler un pont uniquement depuis deux transitions deja observees :
  `mode courant --action--> mode cible --action progressive--> reduction`.
- Autoriser un pont temporairement regressif seulement si le gain compose
  observe est strictement positif ; rejeter les cycles de gain net nul.
- Reprendre un sous-but soutenu apres une courte diversification si une action
  directionnelle compatible redevient disponible.
- Filtrer les candidats sans preuve avant de depenser la rallonge, afin de ne
  pas transformer la persistance en exploration aveugle.
- Auditer dans chaque decision le statut `bridge`, le mode cible, l'action de
  suivi, l'index de tentative et la limite de poursuite.
- Conserver la preuve terminale exclusivement liee aux level-up et WIN observes
  en ligne.
- Fournir une ablation qui conserve tout SAGE.8r et desactive uniquement la
  persistance et la composition de ponts de SAGE.8s.

Ajouts :

- `theory/online_persistent_pursuit.py`
- extension de `theory/online_state_conditioned_effect.py`
- extension de `theory/online_effect_conditioned_subgoal.py`
- extension de `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v10 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_persistent_pursuit.py`
- extension de `tests/test_online_state_conditioned_effect.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8s_persistent_pursuit_ablation.json`
- `diagnostics/sage/sage8s_cn04_persistent_pursuit.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v10`
- `paired_protocol.protocol_gate_passed=true`
- `persistent_directional_pursuit_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.directional_bridge_predictions=0`
- `unified.directional_bridge_selections=0`
- `unified.persistent_pursuit_commitment_selections=0`
- `unified.persistent_pursuit_resumed_commitments=0`
- `unified.persistent_pursuit_attempt_budget_extensions=0`
- `unified.persistent_pursuit_rollout_budget_extensions=0`
- `unified.persistent_pursuit_credit_window_extensions=0`
- `unified.persistent_pursuit_continuation_actions=0`
- `unified.persistent_pursuit_bridge_actions=0`
- `unified.persistent_pursuit_progress_events=0`
- `unified.persistent_pursuit_repeated_progress_events=0`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `persistent_directional_pursuit_enabled_in_unified=false`
- toutes les metriques de poursuite persistante et de pont valent zero ;
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `levels_completed=0`
- `wins=0`

Audit cible `cn04-65d47d14`, seed 0, 2 resets x 40 :

- le sous-but `exhaust(color0)` obtient son premier progres de poursuite ;
- aucune action progressive, contraste de mode inepuise ou pont compose
  strictement rentable n'est ensuite disponible ;
- la garde refuse donc la reprise avant d'etendre la limite, le budget du
  rollout ou la fenetre de credit ;
- aucune action supplementaire inconnue n'est executee ;
- le run conserve exactement les 48 reductions objectives et le progres de
  poursuite de SAGE.8r ;
- `levels_completed=0` et `wins=0`.

Validation synthetique en ligne :

- une transition neutre connue vers un second mode est compilee comme pont ;
- l'action de pont est selectionnee au-dela de la limite initiale ;
- l'action progressive apprise dans le mode cible est ensuite selectionnee ;
- la chaine obtient un second progres sur le meme objectif ;
- le pont et le suivi sont tous deux audites comme actions de la politique
  directionnelle persistante.

Validation :

- `new_sage8s_tests=7 passed`
- `targeted_cognitive_tests=41 passed`
- `full_repository_tests=1395 passed`
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou algorithmique de persistance multi-etapes est franchi. Le
controleur peut maintenant prolonger un sous-but soutenu, planifier un pont de
mode appris et obtenir un nouveau progres sans supervision terminale. Le test
synthetique demontre toute la chaine. Sur les cinq jeux held-out, les deux
sous-buts soutenus par un progres ne disposent toutefois d'aucune continuation
directionnelle rentable qui passe les gardes : aucun engagement persistant
n'est ouvert, aucune rallonge n'est accordee et le resultat est identique a
SAGE.8r. Il ne faut donc presenter cette etape ni comme un gain d'efficacite ni
comme un progres ARC terminal.

Le prochain verrou est l'aliasing des interventions semantiques. Une signature
comme `ACTION6::color8` fusionne encore plusieurs objets et roles spatiaux dont
les effets peuvent etre opposes. SAGE doit apprendre des liaisons
`action x entite x role structurel x mode`, generaliser entre positions
equivalentes sans confondre les instances, puis fournir ces actions ancrees au
planificateur de ponts. C'est cette precision qui manque actuellement pour
produire des continuations rentables sur les niveaux held-out.

## SAGE.8t - entity-anchored semantic interventions

Objectif :

- Remplacer, dans le suffixe causal, l'identite trop large
  `ACTION6::color:N` par une intervention concrete liee a une entite et a son
  role structurel courant.
- Conserver separement une classe de transfert sans coordonnee absolue afin de
  reutiliser une preuve entre positions structurellement equivalentes.
- Distinguer les instances equivalentes par un slot concret pour que la limite
  d'essais, le credit directionnel et les contradictions ne soient plus
  partages aveuglement entre tous les objets d'une meme couleur.
- Combiner cette identite avec le mode latent de SAGE.8r : la cle effective
  devient `action x entite x role structurel x mode`.
- Detecter explicitement les effets opposes de deux instances d'une meme classe
  et demander un contraste d'entite au lieu de transferer arbitrairement la
  premiere preuve.
- Preserver la politique de SAGE.8s tant qu'aucune preuve structurelle ne la
  departage : la signature historique reste le dernier tie-break commun.
- Ne jamais convertir une reduction locale, un transfert ou un contraste
  d'entite en preuve terminale ; seuls level-up et WIN observes en ligne
  conservent cette fonction.

Representation :

- `concrete_signature` encode l'action, la couleur, la forme normalisee, une
  classe d'aire, le role structurel et un slot d'instance.
- `transfer_signature` retire uniquement le slot d'instance ; elle ne contient
  aucune coordonnee absolue.
- Le role structurel encode le contact avec le bord, la relation au joueur, la
  multiplicite de l'entite et ses degres locaux d'adjacence et d'alignement.
- Deux translations absolues d'une meme configuration produisent la meme
  classe de transfert.
- Deux instances equivalentes dans la meme configuration partagent cette
  classe mais conservent des signatures concretes distinctes.
- Une action non cliquable ou un clic non rattachable a un objet conserve sa
  signature historique, sans fabriquer d'ancrage.

Apprentissage et controle :

- La preuve exacte d'une instance dans un mode reste prioritaire.
- En l'absence de preuve exacte, une preuve coherente de la meme classe
  structurelle et du meme mode peut etre transferee.
- Si une instance progresse et une autre regresse dans le meme mode, une
  nouvelle instance recoit `needs_entity_contrast`, jamais `progressive`.
- Une contradiction entre instances dans un meme mode n'est plus comptee comme
  reversibilite modale.
- Les interventions des plans temporels qui participent a une option causale
  sont re-ancrees avant credit ; cela evite de perdre les preuves acquises avant
  l'ouverture explicite du suffixe.
- Les limites d'essais du rollout sont appliquees a la signature concrete, pas
  a toute la couleur.
- Chaque decision audite la classe de transfert, l'entite, le role, le slot,
  l'usage d'un transfert structurel et un conflit d'alias eventuel.

Ajouts :

- `theory/online_semantic_intervention.py`
- extension de `theory/online_state_conditioned_effect.py`
- extension de `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v11 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_semantic_intervention.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8t_entity_anchor_ablation.json`
- `diagnostics/sage/sage8t_cn04_entity_anchors.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v11`
- `paired_protocol.protocol_gate_passed=true`
- `entity_anchored_interventions_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.entity_anchored_candidate_signatures=34`
- `unified.entity_anchored_transfer_signatures=28`
- `unified.entity_anchored_selections=4`
- `unified.directional_entity_anchored_action_models=16`
- `unified.directional_structural_transfer_classes=8`
- `unified.directional_structural_transfer_predictions=6`
- `unified.directional_structural_transfer_selections=2`
- `unified.directional_entity_alias_conflicts=0`
- `unified.directional_entity_contrast_selections=0`
- `unified.causal_option_downstream_actions=12`
- `unified.causal_option_downstream_effects=8`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `entity_anchored_interventions_enabled_in_unified=false`
- toutes les metriques d'ancrage, de transfert structurel et de contraste
  d'entite valent zero ;
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.causal_option_downstream_actions=12`
- `unified.causal_option_downstream_effects=8`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `levels_completed=0`
- `wins=0`

Audit cible `cn04-65d47d14`, seed 0, 2 resets x 40 :

- 12 signatures concretes et 9 classes structurelles sont inventoriees ;
- 8 modeles directionnels sont ancres dans une entite et 4 classes sont
  disponibles pour le transfert ;
- 3 predictions et 1 selection utilisent effectivement une preuve
  structurellement transferee ;
- le clic de poursuite `exhaust(color0)` est maintenant audite avec sa forme,
  son role et son slot au lieu du seul alias `ACTION6::color:8` ;
- le run conserve exactement 61 experiences, 48 reductions objectives, 1
  progres de poursuite et 1 sous-but soutenu comme SAGE.8s ;
- `levels_completed=0` et `wins=0`.

Validation synthetique en ligne :

- deux objets de meme couleur mais de roles differents ne partagent plus leur
  classe de transfert ;
- une translation absolue conserve les signatures structurelles ;
- deux instances equivalentes ont des slots distincts et une classe commune ;
- un progres coherent est transfere a une instance equivalente non testee ;
- des effets opposes produisent `needs_entity_contrast` tandis que les deux
  preuves exactes restent respectivement progressive et regressive ;
- la limite d'essais d'une instance ne censure plus l'autre ;
- l'ablation restaure exactement l'identite couleur de SAGE.8s.

Validation :

- `new_sage8t_tests=8 passed`
- `targeted_cognitive_tests=64 passed`
- `full_repository_tests=1403 passed`
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou de representation semantique est franchi sans regression
held-out. Le controleur sait maintenant quel objet concret a recu une action,
quelle classe structurelle peut porter la preuve ailleurs et quand des
instances equivalentes se contredisent. Les deux selections transferees du run
principal montrent que cette representation est utilisee, mais l'absence de
conflit observe et de continuation persistante signifie qu'elle n'a pas encore
produit de nouvelle competence terminale. Le resultat reste donc neutre sur
ARC : ni niveau ni WIN supplementaire.

Le prochain verrou est l'acquisition active de liaisons causales entre entites.
Les roles de SAGE.8t sont encore des descriptions passives et les slots peuvent
changer quand des objets apparaissent, disparaissent ou se deplacent. SAGE doit
maintenir une correspondance `entite avant -> entite apres`, generer des
contrastes controles entre arguments structurels dans le meme mode, puis
apprendre quelle entite porte l'effet recherche. Cette liaison active doit
alimenter directement les ponts persistants sans utiliser le resultat terminal
pour choisir retrospectivement le bon objet.

## SAGE.8u - active online entity causal bindings

Objectif :

- Suivre l'entite concretement visee entre l'observation avant et apres une
  intervention, y compris lorsqu'elle change de couleur, de forme ou de
  position, et distinguer explicitement stabilite, mouvement, transformation,
  suppression et correspondance ambigue.
- Attribuer une reduction d'objectif a l'argument clique seulement si cette
  entite a elle-meme change ; un progres concomitant avec une cible stable est
  memorise comme `noncarrier_progress`, jamais comme preuve causale positive.
- Conserver un identifiant de piste en ligne entre frames et branches, sans
  utiliser de label terminal ni de representation figee avant l'examen.
- Apprendre des liaisons `option x objectif x mode x argument structurel` et
  distinguer `progressive_carrier`, `regressive_carrier`, `misbound` et
  `needs_controlled_contrast`.
- Generer un contraste entre arguments de meme alias large uniquement apres
  avoir observe, dans le meme mode, un porteur progressif et un autre argument
  regressif ou non porteur.
- Injecter ces contrastes exclusivement dans une poursuite persistante deja
  debloquee par un progres en ligne. Avant ce seuil, SAGE.8u est neutre dans le
  classement des actions et preserve exactement la politique SAGE.8t.
- Ne jamais transformer une liaison locale, une piste d'entite ou un contraste
  controle en preuve terminale : seuls level-up et WIN observes restent des
  preuves terminales.

Representation et apprentissage :

- Le matching combine recouvrement de cellules, forme normalisee, aire,
  couleur et distance. Une transformation de couleur au meme emplacement est
  donc suivie au lieu d'etre decomposee artificiellement en disparition plus
  apparition.
- Une correspondance ambigue n'obtient aucun credit positif ou negatif.
- Une suppression credible compte comme changement de la cible ; une cible
  stable pendant un progres signale que l'effet est porte ailleurs.
- Les preuves restent separees par mode latent et classe structurelle. Le
  conflit requis pour un contraste est constate en ligne entre deux classes,
  pas infere d'un prior ni selectionne retrospectivement avec le resultat du
  niveau.
- Le controle persistant bloque les arguments appris comme regressifs ou mal
  lies, favorise les porteurs progressifs et peut tester un argument
  `needs_controlled_contrast`. Les replays deja soutenus gardent priorite.
- Chaque decision et transition audite le statut de liaison, le gain, la
  confiance, la compatibilite, le conflit, le contraste et l'identifiant de
  piste.

Ajouts :

- `theory/online_entity_causal_binding.py`
- extension de `theory/online_semantic_intervention.py`
- extension de `theory/online_causal_option.py`
- extension de `theory/online_persistent_pursuit.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v12 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_entity_causal_binding.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8u_entity_binding_ablation.json`
- `diagnostics/sage/sage8u_cn04_entity_bindings.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v12`
- `paired_protocol.protocol_gate_passed=true`
- `active_entity_causal_binding_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.entity_binding_observations=6`
- `unified.entity_binding_models=6`
- `unified.entity_binding_matched_entities=6`
- `unified.entity_binding_tracks_created=4`
- `unified.entity_binding_tracks_reused=2`
- `unified.entity_binding_noncarrier_progress_events=2`
- `unified.entity_binding_carrier_progress_events=0`
- `unified.entity_binding_conflicts=0`
- `unified.entity_binding_controlled_contrast_selections=0`
- `unified.causal_option_downstream_actions=12`
- `unified.causal_option_downstream_effects=8`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `active_entity_causal_binding_enabled_in_unified=false`
- toutes les metriques `entity_binding_*` valent zero ;
- les autres metriques agregees sont strictement identiques au run principal,
  notamment 800 actions, 488 experiences, 782 reductions objectives, 2 progres
  de poursuite, 0 niveau et 0 WIN.

Audit cible `cn04-65d47d14`, seed 0, 2 resets x 40 :

- 3 transitions ciblees sont liees a 3 modeles causaux ;
- 2 pistes sont creees et 1 est reutilisee ;
- les 3 cibles sont retrouvees stables ;
- 1 progres objectif est correctement classe `noncarrier_progress` au lieu
  d'etre credite a l'objet clique ;
- aucun conflit de porteur n'est encore observe, donc aucun contraste n'est
  genere ;
- le run conserve exactement 61 experiences, 48 reductions objectives, 1
  progres de poursuite et 1 sous-but soutenu ;
- `levels_completed=0` et `wins=0`.

Validation synthetique en ligne :

- une transformation de couleur et un mouvement reutilisent la meme piste ;
- une suppression est reconnue comme changement de la cible ;
- une correspondance ambigue ne recoit aucun credit causal ;
- un progres avec cible stable produit `misbound` ;
- un porteur progressif oppose a un argument non porteur declenche un contraste
  controle dans le meme mode ;
- ce contraste guide le suffixe uniquement pendant une poursuite persistante ;
- l'ablation ne conserve aucune observation, prediction ou selection SAGE.8u.

Validation :

- `new_sage8u_tests=9 passed`
- `targeted_cognitive_tests=44 passed`
- `full_repository_tests=1412 passed` (1405 groupes ensemble et les 7 tests
  d'un fichier historique a namespace collisionne executes isolement)
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou de liaison active entre intervention et entite visee est
franchi sans regression held-out. Le resultat apporte surtout une information
negative utile : sur les six transitions ciblees, aucun progres n'est porte par
l'objet clique et deux progres se produisent pendant que cette cible reste
stable. SAGE evite donc maintenant un faux credit que SAGE.8t ne pouvait pas
detecter. En l'absence d'un porteur positif oppose, la garde en ligne interdit
correctement de fabriquer un contraste, et le resultat ARC reste neutre.

Le prochain verrou est la localisation du porteur indirect de l'effet. SAGE.8u
sait que l'objet clique n'a pas porte certains progres, mais ne suit pas encore
toutes les autres entites et relations modifiees pour identifier laquelle a
porte le changement. La prochaine etape doit construire en ligne des hyperaretes
`entite actionnee -> relation -> entite affectee`, departager les porteurs
indirects par interventions controlees, puis fournir ces effets mediatises au
planificateur persistant sans aucun label terminal retrospectif.

## SAGE.8v - mediated entity-effect induction

Objectif :

- Etendre la correspondance SAGE.8u de l'objet clique a toutes les entites de
  la scene avant et apres chaque intervention ancree.
- Distinguer pour chaque piste stabilite, mouvement, transformation,
  suppression, apparition et ambiguite dans un matching bijectif.
- Decrire chaque entite indirectement affectee par une hyperarete sans
  coordonnee absolue : `entite actionnee -> relation -> entite affectee`.
- Former un ensemble de porteurs candidats lorsqu'un objectif progresse alors
  que la cible cliquee reste stable.
- Departager les changements concomitants par intersection de repetitions
  controlees : un candidat qui n'est pas commun aux contextes progressifs ou
  qui apparait autant dans les controles sans progres est elimine.
- Exiger deux contextes progressifs concordants avant le statut `supported`.
  Une observation unique ne produit que `needs_mediator_contrast`.
- Autoriser une preuve mediatisee a expliquer une cible SAGE.8u `misbound`,
  sans recrediter cette cible et uniquement dans une poursuite deja soutenue
  par un progres en ligne.
- Conserver la preuve terminale strictement separee : aucune correspondance,
  hyperarete ou reduction locale ne remplace un level-up ou WIN observe.

Representation et controle :

- Le matching de scene combine recouvrement, forme normalisee, aire, couleur
  et distance, puis impose une affectation un-a-un.
- Les pistes sont propagees aux entites retrouvees et reutilisees entre frames.
  Une apparition recoit une nouvelle piste.
- Une ambiguite sur une source ou une cible liee n'est jamais reinterpretee
  comme suppression plus apparition et ne recoit aucun credit de porteur.
- La signature du porteur encode son changement, son entite, son role
  structurel et sa relation a l'entite actionnee : proximite, direction,
  alignement et relation de couleur.
- Les modeles sont separes par option, objectif, mode latent et classe de
  transfert de l'action cible.
- `needs_mediator_contrast` peut demander une repetition bornee ; `supported`
  peut guider une continuation ; `contradicted` bloque l'action en poursuite
  persistante. Avant la persistance, tous ces rangs restent neutres.
- Chaque decision et transition audite le statut mediatise, le gain, les
  candidats, le porteur soutenu et le nombre de changements de scene.

Ajouts :

- `theory/online_mediated_entity_effect.py`
- extension de `theory/online_causal_option.py`
- extension de `theory/online_persistent_pursuit.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v13 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_mediated_entity_effect.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8v_mediated_effect_ablation.json`
- `diagnostics/sage/sage8v_cn04_mediated_effects.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v13`
- `paired_protocol.protocol_gate_passed=true`
- `mediated_entity_effect_induction_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.experiment_cost_rate=0.61`
- `unified.mediated_effect_observations=6`
- `unified.mediated_effect_scene_correspondences=48`
- `unified.mediated_effect_changed_entities=6`
- `unified.mediated_effect_transformed_entities=6`
- `unified.mediated_effect_tracks_created=32`
- `unified.mediated_effect_tracks_reused=16`
- `unified.mediated_effect_progress_with_indirect_candidates=2`
- `unified.mediated_effect_direct_target_progress_events=0`
- `unified.mediated_effect_models=6`
- `unified.mediated_effect_supported_hyperedges=0`
- `unified.mediated_effect_controlled_contrast_predictions=0`
- `unified.mediated_effect_controlled_contrast_selections=0`
- `unified.effect_conditioned_pursuit_progress_events=2`
- `unified.progress_supported_effect_conditioned_subgoals=2`
- `unified.objective_distance_reductions=782`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Ablation complete, memes jeux, seeds, resets et budgets :

- `mediated_entity_effect_induction_enabled_in_unified=false`
- toutes les metriques `mediated_effect_*` valent zero ;
- les metriques de politique mediatisee persistante valent zero ;
- toutes les autres metriques agregees sont strictement identiques au run
  principal : aucune action, experience, reduction, progression locale ou
  issue terminale ne diverge.

Audit cible `cn04-65d47d14`, seed 0, 2 resets x 40 :

- 3 interventions produisent 24 correspondances de scene ;
- 16 pistes sont creees et 8 sont reutilisees ;
- 3 entites indirectes sont transformees tandis que les cibles cliquees
  restent stables ;
- le progres `exhaust(color0)` est relie a une transformation d'une structure
  color0 adjacente, au-dessus et a droite de l'entite actionnee ;
- cette hyperarete reste `needs_mediator_contrast` car elle n'a qu'un contexte
  progressif ;
- aucun porteur n'est donc promu ni selectionne ;
- le run conserve exactement 61 experiences, 48 reductions objectives, 1
  progres de poursuite et 1 sous-but soutenu ;
- `levels_completed=0` et `wins=0`.

Validation synthetique en ligne :

- une entite indirecte deplacee conserve sa piste sur plusieurs frames ;
- transformations, suppressions et apparitions sont distinguees ;
- un premier progres demande un contraste de mediateur ;
- deux repetitions traduites mais structurellement equivalentes soutiennent
  la meme hyperarete ;
- l'intersection de deux ensembles elimine un changement concomitant ;
- une correspondance ambigue ne cree aucun porteur candidat ;
- un contraste mediatise peut poursuivre une action dont la cible est
  correctement classee `misbound` ;
- l'ablation ne conserve ni observation, ni prediction, ni hyperarete.

Validation :

- `new_sage8v_tests=10 passed`
- `targeted_cognitive_tests=54 passed`
- `full_repository_tests=1422 passed` (1415 groupes ensemble et les 7 tests
  du fichier historique a namespace collisionne executes isolement)
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou de localisation du porteur indirect est franchi sans
regression held-out. SAGE transforme maintenant les deux `noncarrier_progress`
de SAGE.8u en deux ensembles de candidats explicites et position-invariants.
Sur `cn04`, le candidat est une structure color0 transformee, et non l'objet
color8 clique. La garde de confirmation empeche toutefois de promouvoir cette
correlation unique : aucune competence terminale supplementaire n'est encore
revendiquee.

Le prochain verrou est l'acquisition active d'un second contexte pour ces
hyperaretes. Le controleur sait demander `needs_mediator_contrast` dans un
suffixe persistant, mais le candidat apparait au dernier segment du rollout et
n'est pas automatiquement repropose apres une nouvelle ouverture ou branche.
La prochaine etape doit reserver et ordonnancer une replication
contre-factuelle inter-branche du meme mecanisme, reouvrir le contexte causal,
varier un seul argument relationnel, puis confirmer ou refuter le mediateur
avant toute exploitation terminale.

## SAGE.8w - active cross-branch mediated replication

Objectif :

- Transformer chaque `needs_mediator_contrast` productif en requete
  experimentale persistante plutot qu'en simple bonus local de selection.
- Interdire la repetition dans la branche source et attendre une branche
  independante.
- Si l'ouverture ne reapparait pas spontanement, reserver un nouveau demarrage
  du plan causal de preparation correspondant, meme lorsque son budget normal
  de demarrages est epuise.
- Activer la requete uniquement apres une ouverture reellement observee de la
  meme option.
- Rejouer exactement la meme classe semantique d'intervention, tout en laissant
  varier le contexte relationnel et les entites indirectement affectees.
- Donner priorite a cette replication sur les predictions directionnelles ou
  liaisons de cible contradictoires : celles-ci ne doivent pas bloquer
  l'experience qui est precisement destinee a les departager.
- Confirmer, refuter ou replanifier exclusivement depuis le gain objectif local
  et les correspondances de scene du nouveau contexte. Aucun etat terminal
  n'entre dans la resolution.
- Borner les repetitions non concluantes et exposer une ablation isolee.

Representation et controle :

- Une requete memorise option, arete causale, objectif, sous-but, mode latent,
  classe de transfert de l'action, branche et contexte source, ainsi que
  l'ensemble initial de porteurs candidats.
- `pending` reserve une preparation sur une branche ulterieure ; `active`
  signifie que l'option vient d'etre rouverte ; `confirmed`, `refuted` et
  `expired` ferment la requete.
- Le planificateur temporel accepte une arete causale preferee pour SAGE.8w.
  Cette reservation peut depasser le budget normal du plan, mais reste soumise
  aux gardes mesurables de chaque etape et a une seule action primitive suivie
  d'une reobservation.
- La selection du suffixe impose la meme signature de transfert semantique,
  sans coordonnee absolue ni identite d'instance.
- La decision et la transition auditent l'identifiant de requete, la branche,
  la replication exacte, les preparations, activations, selections et
  resolutions.

Ajouts :

- `theory/online_mediated_replication.py`
- extension de `theory/online_causal_option.py`
- reservation de preparation dans `theory/online_temporal_goal_composition.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema et ablation v14 dans `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_mediated_replication.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/unified_cognition_ab_held_out.json`
- `diagnostics/sage/sage8w_cn04_active_replication.json`
- `diagnostics/sage/sage8w_active_replication_ablation.json`

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v14`
- `paired_protocol.protocol_gate_passed=true`
- `active_mediated_replication_enabled_in_unified=true`
- `unified.controller_errors=0`
- `unified.actions_executed=800`
- `unified.experiment_actions=488`
- `unified.objective_distance_reductions=782`
- `unified.mediated_replication_requests_created=2`
- `unified.mediated_replication_pending_requests=2`
- `unified.mediated_replication_preparation_starts=0`
- `unified.mediated_replication_cross_branch_activations=0`
- `unified.mediated_replication_selections=0`
- `unified.mediated_replication_confirmations=0`
- `unified.mediated_replication_refutations=0`
- `legacy_only.levels_completed=0`
- `unified.levels_completed=0`
- `legacy_only.wins=0`
- `unified.wins=0`
- `arc_progress_observed=false`

Le protocole historique a deux resets produit les candidats a la fin de la
derniere branche ; il prouve donc la creation et la persistance des requetes,
mais ne contient pas de branche ulterieure pour les executer. Les 800 actions,
488 experiences, 782 reductions locales et issues terminales restent
strictement identiques a SAGE.8v.

Audit actif cible `cn04-65d47d14`, seed 0, 4 resets x 40 :

- 1 requete est creee en branche 2 pour la transformation d'une structure
  color0 apres une intervention sur la structure color8 ;
- en branche 3, SAGE.8w reserve 1 nouveau demarrage du plan causal et execute
  3 actions de preparation ;
- la meme option est effectivement rouverte dans ce contexte independant ;
- 4 predictions de replication exacte sont produites et 1 action reservee est
  selectionnee ;
- le second progres transforme trois porteurs possibles : une cellule color0,
  une structure color12 et une structure color4 ;
- aucun de ces trois porteurs ne partage la signature complete du candidat
  color0 initial ; la requete est donc `refuted` et non promue ;
- `mediated_replication_confirmations=0` ;
- `objective_distance_reductions=136`, contre 135 dans l'ablation ;
- `levels_completed=0` et `wins=0` dans les deux bras.

Ablation cible, memes jeu, seed, resets et budgets :

- `active_mediated_replication_enabled_in_unified=false` ;
- toutes les metriques `mediated_replication_*` valent zero ;
- le modele SAGE.8v reste actif, mais son candidat unique n'entraine ni
  preparation, ni reouverture, ni seconde intervention ;
- le run execute 160 actions, 68 experiences et 135 reductions objectives,
  sans niveau ni WIN.

Validation synthetique en ligne :

- une ambiguite productive cree une seule requete persistante ;
- la branche source est explicitement bloquee ;
- une branche ulterieure reserve la preparation et active la requete seulement
  apres une vraie reouverture ;
- seule la meme classe semantique d'intervention est compatible ;
- une deuxieme observation peut confirmer sans label terminal ;
- un ensemble incompatible refute la requete ;
- des repetitions toujours ambigues expirent apres un budget borne ;
- la replication reservee domine les autres rangs du suffixe ;
- l'ablation ne conserve aucune requete, preparation ou resolution.

Validation :

- `new_sage8w_tests=8 passed`
- `targeted_cognitive_tests=50 passed`
- `full_repository_tests=1430 passed` (1423 groupes ensemble et les 7 tests
  du fichier historique a namespace collisionne executes isolement)
- `scoped_ruff_and_compileall=passed`

Lecture : le verrou d'acquisition active est franchi. SAGE ne se contente plus
de savoir qu'une hyperarete manque de preuve : il recree le contexte causal,
reexecute l'intervention et tranche en ligne. Sur `cn04`, cette etape evite une
fausse promotion et apporte une preuve negative nouvelle, mais aucun niveau
ARC-AGI-3 supplementaire.

Le prochain verrou est l'anti-unification structurelle en ligne des porteurs.
La refutation SAGE.8w montre que la signature SAGE.8v est trop specifique : le
premier et le second contexte contiennent tous deux un porteur color0
transforme, mais sa forme, son role et sa relation concrete changent. La
prochaine etape doit construire apres chaque transition une petite lattice
d'abstractions des hyperaretes, conserver le niveau le plus precis compatible
avec les contextes progressifs, l'opposer aux controles sans progres, puis
selectionner de nouvelles interventions qui discriminent ces abstractions sans
jamais figer la representation avant l'examen.

## SAGE.8x - online structural anti-unification of mediated carriers

Objectif :

- Ne plus refuter automatiquement un mecanisme lorsque deux transitions
  progressives affectent des instances de forme, role ou relation differents.
- Construire la representation commune uniquement depuis les observations
  acquises pendant l'episode, sans catalogue de reponses ni structure figee
  avant l'evaluation.
- Conserver le motif structurel le plus specifique compatible avec tous les
  contextes progressifs et relie a l'objectif courant.
- Opposer chaque motif aux transitions sans progres et regressives ; deux
  supports independants sont requis, un contexte regressif le veto et des
  controles aussi nombreux que les supports empechent sa promotion.
- Preserver le comportement exact SAGE.8w derriere une ablation isolee.

Representation et controle :

- Chaque signature descriptive SAGE.8v est decomposee en attributs generiques
  de changement, couleur, forme, aire, position, multiplicite, role structural
  et relation.
- L'intersection de chaque nouveau contexte progressif construit en ligne une
  petite lattice bornee a 64 motifs. Un motif doit conserver le type de
  changement et au moins une identite causale parmi couleur, forme ou aire.
- Lorsque chaque contexte contient un candidat portant une couleur de
  l'objectif, la lattice est ancree sur ces candidats ; elle ne peut donc pas
  preferer une structure concomitante plus detaillee mais sans rapport avec le
  but poursuivi.
- Les motifs sont classes par ancrage objectif, specificite ponderee et marge
  supports-controles. Une egalite au meilleur rang reste ambigue.
- Une hyperarete abstraite soutenue est exposee dans les predictions et les
  decisions exactement comme une hyperarete concrete, avec ses attributs et sa
  specificite auditables.
- Le chemin complet propage cette preuve a travers le magasin d'options, le
  controleur unifie, les transitions, le resume et le benchmark A/B v15.

Ajouts :

- `theory/online_mediated_abstraction.py`
- extension de `theory/online_mediated_entity_effect.py`
- integration dans `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema, metriques et ablation v15 dans
  `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_mediated_abstraction.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/sage8x_cn04_mediated_abstraction.json`
- `diagnostics/sage/sage8x_mediated_abstraction_ablation.json`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Audit actif cible `cn04-65d47d14`, seed 0, 4 resets x 40 :

- les deux contextes progressifs ne partagent aucune signature concrete ;
- la lattice apprend apres observation un porteur `transformed`, `color0`,
  `area9+`, `interior`, `unique`, adjacent et decale, sans imposer la forme ni
  l'orientation verticale/horizontale qui changent entre les contextes ;
- `mediated_abstraction_hypotheses=1` ;
- `mediated_abstraction_supported_hyperedges=1` ;
- `mediated_effect_supported_hyperedges=1`, contre 0 avec l'ablation ;
- `mediated_abstraction_control_contexts=0` et
  `mediated_abstraction_regression_contexts=0` : aucun controle observe ne
  satisfait le motif appris ;
- les 160 actions, 73 experiences et 136 reductions objectives sont identiques
  dans les deux bras, ce qui isole la nouvelle connaissance de toute variation
  de trajectoire ;
- la refutation de replication SAGE.8w reste valide : elle concerne un autre
  mode latent regressif (`exhaust`, distance 1), tandis que l'abstraction est
  soutenue dans le mode `exhaust`, distance 2 ;
- `levels_completed=0` et `wins=0` dans les deux bras.

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v15` ;
- `paired_protocol.protocol_gate_passed=true` ;
- `online_mediated_anti_unification_enabled_in_unified=true` ;
- `unified.controller_errors=0` ;
- `unified.actions_executed=800` ;
- `unified.experiment_actions=488` ;
- `unified.objective_distance_reductions=782` ;
- les 2 requetes SAGE.8w restent en attente a la fin du dernier reset ;
- aucune paire n'offre deux contextes progressifs pour le meme mecanisme : les
  quatre metriques d'abstraction restent donc a zero et la politique est
  strictement identique a SAGE.8w ;
- `legacy_only.levels_completed=0`, `unified.levels_completed=0`,
  `legacy_only.wins=0` et `unified.wins=0`.

Validation synthetique en ligne :

- le parseur expose tous les attributs generiques d'un porteur ;
- l'anti-unification elimine forme, role et relation non invariants ;
- l'ancrage objectif evite un candidat concomitant de mauvaise couleur ;
- deux controles correspondants bloquent la promotion ;
- un seul contexte regressif correspondant oppose un veto ;
- l'evidence mediee promeut l'hyperarete abstraite lorsque l'intersection exacte
  est vide ;
- l'ablation conserve la refutation exacte SAGE.8w.

Validation :

- `targeted_cognitive_tests=50 passed` ;
- `full_repository_tests=1438 passed` (1431 groupes ensemble et les 7 tests du
  fichier historique a namespace collisionne executes isolement) ;
- `scoped_ruff_and_compileall=passed`.

Lecture : le verrou de representation est franchi. SAGE sait maintenant
decouvrir pendant l'examen qu'un meme mecanisme peut etre porte par des
instances structurellement differentes et ne generalise que ce qui survit aux
observations positives, controles et regressions. Sur `cn04`, cette etape
remplace une fausse refutation concrete par une hyperarete abstraite soutenue,
mais elle est apprise au dernier contexte disponible et ne produit encore
aucun niveau ARC-AGI-3 supplementaire.

Le prochain verrou est la discrimination active de la lattice. SAGE doit
transformer les attributs encore presents dans le motif en interventions
contre-factuelles appariees, faire varier une propriete a la fois, reserver une
nouvelle ouverture inter-branche meme lorsqu'un motif est deja soutenu, puis
confirmer sa necessite ou l'eliminer exclusivement depuis le progres terminal
en ligne. Ce n'est qu'apres ces controles actifs que l'hyperarete abstraite
doit devenir une politique composee vers le but terminal.

## SAGE.8y - active one-feature discrimination of mediated abstractions

Objectif :

- Transformer chaque hyperarete abstraite soutenue en une experience causale
  persistante plutot qu'en une conclusion definitive.
- Rechercher en ligne un porteur qui differe du motif appris par exactement un
  attribut, reouvrir la meme option dans une branche ulterieure et rejouer la
  meme classe semantique d'intervention.
- Eliminer l'attribut si le contraste conserve le progres, ou le declarer
  necessaire si le progres disparait, exclusivement depuis la variation de
  distance a l'objectif observee apres l'action.
- Refuser toute conclusion lorsque le mode latent, l'intervention, plusieurs
  attributs ou la correspondance du porteur varient simultanement.

Representation et controle :

- `OnlineMediatedDiscriminationStore` conserve une requete par hyperarete,
  l'attribut teste, la valeur de controle attendue, la branche source, la
  branche active, le mode latent et un budget d'essais borne.
- Les attributs de la lattice sont proposes dans un ordre structurel fixe ; ce
  n'est pas une reponse de niveau mais l'ordre generique des variables a
  falsifier.
- Une prediction n'est admissible que pour la meme classe de transfert
  d'action, une branche posterieure, exactement le meme mode latent et une
  signature prospective qui differe par une seule propriete.
- Un progres avec contraste observe elimine la propriete. Une absence de
  progres sur objectif stable et correspondance non ambigue la rend
  necessaire. Tous les autres resultats restent inconclusifs.
- La requete reserve la preparation et la reouverture de l'option meme lorsque
  l'hyperarete SAGE.8x est deja soutenue, puis domine les autres choix du
  suffixe lorsqu'un contraste valide existe.
- Le chemin complet propage la requete, la preparation, la prediction, la
  selection et la resolution a travers l'option causale, le plan temporel, le
  controleur unifie et le benchmark A/B v16.

Ajouts :

- `theory/online_mediated_discrimination.py`
- signatures prospectives publiques dans
  `theory/online_mediated_entity_effect.py`
- integration dans `theory/online_causal_option.py`
- reservation dans `theory/online_temporal_goal_composition.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema, metriques et ablation v16 dans
  `theory/unified_cognition_ab_benchmark.py`
- `tests/test_online_mediated_discrimination.py`
- extension de `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/sage8y_cn04_lattice_discrimination.json`
- `diagnostics/sage/sage8y_lattice_discrimination_ablation.json`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Audit actif cible `cn04-65d47d14`, seed 0, 5 resets x 40 :

- SAGE.8x produit 1 hypothese abstraite et 1 hyperarete soutenue ;
- SAGE.8y cree 1 requete, la reactive dans 2 branches ulterieures et reserve 2
  preparations, soit 6 actions de preparation ;
- 12 candidats sont detectes apres reouverture, mais leur mode latent differe
  du mode source (`exhaust`, distance 2) ;
- ces 12 contrastes confondus sont correctement bloques : aucune prediction,
  selection, exigence ou elimination d'attribut n'est enregistree ;
- `controller_errors=0`, `levels_completed=0` et `wins=0` ;
- le bras actif execute 200 actions, 87 experiences et 164 reductions
  objectives, contre 200 actions, 76 experiences et 180 reductions avec
  l'ablation ; l'acquisition active coute donc 11 experiences et 16 reductions
  locales sur ce run sans gain terminal.

Ablation cible, memes jeu, seed, resets et budgets :

- `active_mediated_discrimination_enabled_in_unified=false` ;
- toutes les metriques `mediated_discrimination_*` valent zero ;
- SAGE.8x apprend toujours la meme hyperarete, ce qui isole la nouvelle
  politique d'experimentation de la representation precedente.

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v16` ;
- `paired_protocol.protocol_gate_passed=true` ;
- `active_mediated_discrimination_enabled_in_unified=true` ;
- `unified.controller_errors=0` ;
- `unified.actions_executed=800` ;
- `unified.experiment_actions=488` ;
- `unified.objective_distance_reductions=782` ;
- aucune paire n'atteint les deux supports necessaires a SAGE.8x ; toutes les
  metriques SAGE.8y restent donc a zero et la trajectoire demeure identique a
  SAGE.8x ;
- `legacy_only.levels_completed=0`, `unified.levels_completed=0`,
  `legacy_only.wins=0` et `unified.wins=0`.

Validation synthetique en ligne :

- la branche source, une autre intervention et un autre mode latent sont
  bloques ;
- une variation d'un seul attribut produit une prediction ;
- deux variations simultanees sont refusees ;
- le progres elimine l'attribut teste ;
- l'absence de progres sur correspondance stable le rend necessaire ;
- une correspondance ambigue ne peut produire aucune fausse necessite ;
- apres resolution, le prochain attribut de la lattice est propose ;
- l'ablation ne conserve aucune requete ni prediction.

Validation :

- `new_sage8y_tests=8 passed` ;
- `targeted_cognitive_tests=67 passed` ;
- `full_repository_tests=1447 passed` (1440 groupes ensemble et les 7 tests du
  fichier historique a namespace collisionne executes isolement) ;
- `scoped_ruff_and_compileall=passed`.

Lecture : le verrou de formulation et de controle des contrastes actifs est
franchi. SAGE sait convertir une abstraction apprise pendant l'examen en une
question causale falsifiable et refuse une conclusion quand l'experience ne
fait pas varier une seule variable. Aucun niveau ARC-AGI-3 supplementaire
n'est encore gagne.

Le prochain verrou est la synthese de contexte contre-factuel. Reouvrir la meme
option ne suffit pas : le planificateur doit restaurer le meme mode latent
(`exhaust`, distance 2 sur `cn04`) tout en faisant varier un seul attribut du
porteur. Il faut donc apprendre en ligne des actions de restauration d'etat,
les composer avant l'intervention discriminante et verifier que les autres
variables causales sont demeurees invariantes.

## SAGE.8z - learned latent-mode restoration before causal contrast

Objectif :

- Ne plus attendre passivement qu'une reouverture retrouve le mode latent de
  l'hyperarete abstraite.
- Apprendre uniquement depuis les transitions executees quelles actions font
  passer d'un mode latent a un autre, puis composer un chemin borne vers le
  mode exact exige par la requete SAGE.8y.
- Revalider le mode apres chaque action et ne lancer le contraste a une
  propriete qu'apres restauration effectivement observee.
- Conserver une mesure locale utile meme si son interpretation comme but
  terminal a ete refutee ; cette reservation ne repromeut jamais le but comme
  terminal.
- Preserver exactement SAGE.8y derriere une ablation isolee.

Representation et controle :

- Chaque preuve directionnelle conserve deja le mode avant, la classe
  semantique de l'action et les modes apres observes. SAGE.8z transforme ces
  preuves en graphe de transitions deterministes propre a l'option et a la
  mesure objective courantes.
- Une arete est admissible seulement si elle a ete observee, n'a produit aucun
  echec dangereux et conduit toujours au meme mode. Une transition ambigue ou
  non deterministe est rejetee.
- Une recherche en largeur borne les recettes a trois actions. La premiere
  action concrete doit exister dans la scene actuelle ; chaque etape suivante
  est recalculee apres observation, sans simulation de l'environnement.
- Une action localement regressive peut etre selectionnee lorsqu'elle est une
  etape observee vers le contexte experimental demande. Elle domine alors les
  politiques directionnelles ordinaires, mais seulement pour la requete
  active et avec trois actions de restauration maximum par branche.
- Le sous-but de mesure SAGE.8y est reserve pendant la restauration, meme si sa
  conjecture terminale a ete refutee. Cette exception est active seulement
  avec SAGE.8z et disparait dans l'ablation.
- Le controleur expose une source distincte
  `causal_option_mode_restoration`, le mode cible, le prochain mode attendu,
  la longueur du chemin et sa confiance. La transition reelle confirme ou
  invalide ensuite chaque etape.
- Le benchmark A/B v17 mesure actions, predictions, selections, etapes
  confirmees, modes cibles atteints, echecs et contextes sans recette.

Ajouts :

- composition de restauration dans
  `theory/online_state_conditioned_effect.py`
- reservation de mesure locale dans
  `theory/online_effect_conditioned_subgoal.py`
- etat, budgets et resolution de restauration dans
  `theory/online_mediated_discrimination.py`
- integration dans `theory/online_causal_option.py`
- integration et audit dans `theory/unified_cognitive_controller.py`
- schema, metriques, CLI et ablation v17 dans
  `theory/unified_cognition_ab_benchmark.py`
- extensions de `tests/test_online_state_conditioned_effect.py`,
  `tests/test_online_effect_conditioned_subgoal.py`,
  `tests/test_online_mediated_discrimination.py` et
  `tests/test_unified_cognition_ab_benchmark.py`
- `diagnostics/sage/sage8z_cn04_mode_restoration.json`
- `diagnostics/sage/sage8z_mode_restoration_ablation.json`
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`

Audit actif cible `cn04-65d47d14`, seed 0, 5 resets x 40 :

- la requete exige le mode `exhaust`, distance 2, mais les deux nouvelles
  ouvertures commencent en distance 1 ;
- SAGE retrouve dans sa memoire un chemin observe compose de `ACTION2`, puis
  de l'intervention semantique `ACTION6` sur l'instance structurelle deja
  reliee au passage distance 1 vers distance 2 ;
- `mediated_restoration_predictions=5` ;
- `mediated_restoration_selections=4` et
  `mediated_restoration_actions=4` ;
- les 4 etapes produisent exactement le mode attendu, sans echec, et le mode
  cible est atteint dans 2 branches independantes ;
- apres restauration, 2 predictions de contraste sont produites et 1 action
  est selectionnee ;
- cette action varie seulement `proximity` et son absence de progres en
  controle apparie enregistre cette propriete comme requise dans le modele
  courant ;
- une deuxieme requete est creee pour poursuivre la lattice ;
- le bras actif execute 200 actions, 86 experiences et 180 reductions
  objectives, avec `controller_errors=0`, `levels_completed=0` et `wins=0`.

Ablation cible, memes jeu, seed, resets et budgets :

- `active_mode_restoration_enabled_in_unified=false` ;
- toutes les metriques `mediated_restoration_*` valent zero ;
- le comportement SAGE.8y est reproduit : 87 experiences, 164 reductions,
  aucune prediction ni selection discriminante ;
- SAGE.8z apporte donc 16 reductions objectives supplementaires avec une
  experience de moins sur ce run, mais aucun gain terminal.

Run principal du 2026-07-20, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v17` ;
- `paired_protocol.protocol_gate_passed=true` ;
- `active_mode_restoration_enabled_in_unified=true` ;
- `unified.controller_errors=0` ;
- `unified.actions_executed=800` ;
- `unified.experiment_actions=488` ;
- `unified.objective_distance_reductions=782` ;
- aucune requete SAGE.8y n'est formee dans ces 2 resets : toutes les metriques
  de restauration valent zero et la trajectoire reste identique a SAGE.8y ;
- `legacy_only.levels_completed=0`, `unified.levels_completed=0`,
  `legacy_only.wins=0` et `unified.wins=0`.

Validation synthetique en ligne :

- une action regressive observee peut restaurer directement le mode cible ;
- deux transitions observees sont composees en une recette de longueur 2 ;
- une transition menant a plusieurs modes est refusee ;
- chaque etape confirmee et chaque mode cible atteint sont audites ;
- la restauration dispose d'une ablation independante ;
- un sous-but progressif reserve reste mesurable apres refutation de sa valeur
  terminale, sans modifier ce statut terminal ;
- l'ablation reproduit les compteurs et la trajectoire SAGE.8y.

Validation :

- `targeted_cognitive_tests=57 passed` ;
- `full_repository_tests=1454 passed` (1447 groupes ensemble et les 7 tests du
  fichier historique a namespace collisionne executes isolement) ;
- `scoped_ruff_and_compileall=passed`.

Lecture : le verrou de reconstruction volontaire du contexte est franchi sur
le cas actif. Pour la premiere fois dans cette chaine, la nouvelle capacite
convertit aussi son cout en amelioration A/B locale et declenche l'experience
causale qu'elle devait rendre possible. Aucun niveau ARC-AGI-3 supplementaire
n'est toutefois gagne.

Le prochain verrou est l'exploitation terminale de la lattice revisee. Apres
avoir appris qu'une propriete du porteur est requise, SAGE doit compiler les
contraintes encore soutenues avec la recette de restauration et l'intervention
productive en une politique persistante, l'executer jusqu'a une transition de
niveau et ne lui attribuer une valeur terminale qu'apres ce resultat reel.

## SAGE.9 - online terminal exploitation of the revised carrier lattice

Objectif :

- Ne plus laisser une discrimination SAGE.8y resolue sans effet sur le
  controle : compiler immediatement son resultat en politique executable.
- Conserver les proprietes requises, retirer les proprietes eliminees et
  n'autoriser que la meme classe semantique d'intervention dans le meme mode
  latent appris.
- Reouvrir cette politique sur les branches suivantes, restaurer son mode avec
  les recettes SAGE.8z et lui donner priorite sur la question suivante de la
  lattice.
- Borner les essais par action et par branche, refuter sur deux branches de
  non-progres independantes ou au premier echec dangereux.
- Distinguer strictement support de progres et support terminal : une reduction
  de distance soutient la politique, mais seul un changement de niveau ou WIN
  observe peut lui attribuer une valeur terminale.
- Preserver exactement SAGE.8z derriere une ablation isolee.

Representation et controle :

- `OnlineMediatedExploitationStore` recoit uniquement une requete de
  discrimination resolue `feature_required` ou `feature_eliminated`. Sa cle
  contient option, objectif, mode latent, transfert semantique et abstraction ;
  aucune identite de niveau ni trajectoire-reponse n'est stockee.
- Une revision de la meme cle met a jour les ensembles de proprietes requises
  et eliminees au lieu de creer une competence concurrente.
- La prediction reconstruit les porteurs prospectifs de la scene courante et
  exige une correspondance exacte sur toutes les proprietes effectives. Une
  action concrete deja essayee dans la branche ne peut pas boucler.
- Les statuts sont `candidate`, `progress_supported`, `terminal_supported`,
  `refuted` et `expired`. Le statut terminal ne depend jamais de la distance
  locale ni du statut terminal suppose du sous-but.
- La preparation temporelle, la reservation du sous-but mesurable et la
  restauration de mode sont persistantes entre branches. Les decisions sont
  exposees sous `causal_option_mediated_exploitation` et
  `causal_option_mediated_exploitation_restoration`.
- Une seule cause de controle recoit le credit d'une action : l'exploitation
  domine sa restauration, puis SAGE.8z, SAGE.8y et SAGE.8w. Cela evite le double
  comptage d'une meme transition comme restauration de deux experiences.
- Le benchmark A/B v18 expose compilation, revisions, activations, predictions,
  selections, progres, non-progres, support terminal, refutations, preparation
  et restauration ; `--disable-terminal-mediated-exploitation` isole SAGE.9.

Ajouts :

- nouvelle politique dans `theory/online_mediated_exploitation.py` ;
- integration, priorite, restauration, budgets et credit dans
  `theory/online_causal_option.py` ;
- reservation de preparation dans
  `theory/online_temporal_goal_composition.py` ;
- configuration et audit dans `theory/unified_cognitive_controller.py` ;
- schema, metriques, CLI et ablation v18 dans
  `theory/unified_cognition_ab_benchmark.py` ;
- 11 tests synthetiques dans
  `tests/test_online_mediated_exploitation.py` et extension du benchmark ;
- `diagnostics/sage/sage9_cn04_terminal_exploitation.json` ;
- `diagnostics/sage/sage9_terminal_exploitation_ablation.json` ;
- mise a jour de `diagnostics/sage/unified_cognition_ab_held_out.json`.

Audit actif cible `cn04-65d47d14`, seed 0, 5 resets x 40 :

- la conclusion SAGE.8z `proximity=adjacent` compile exactement 1 politique ;
- la politique est activee 2 fois, dont une reouverture preparee sur une branche
  ulterieure ;
- 2 actions de restauration sont selectionnees, les 2 transitions attendues
  sont confirmees et le mode cible est atteint ;
- 2 predictions d'exploitation sont produites, 1 action est selectionnee et
  cette action produit 1 progres objectif avec le porteur conforme observe ;
- aucun non-progres fiable, echec dangereux ou refutation n'est enregistre ;
- `experiment_actions=84`, contre 86 dans l'ablation ;
- `objective_distance_reductions=180` dans les deux bras : SAGE.9 conserve le
  gain local de SAGE.8z avec deux experiences de moins ;
- `controller_errors=0`, `levels_completed=0`, `wins=0` et donc aucun faux
  support terminal.

Ablation cible, memes jeu, seed, resets et budgets :

- `terminal_mediated_exploitation_enabled_in_unified=false` ;
- toutes les metriques `mediated_exploitation_*` valent zero ;
- la trajectoire SAGE.8z est reproduite : 86 experiences, 180 reductions,
  4 actions de restauration discriminante et aucune politique compilee ;
- aucun niveau ni WIN n'est obtenu.

Run principal du 2026-07-21, memes 5 jeux public-unseen, seeds 0/1,
2 resets, 40 actions par reset :

- `schema_version=sage.unified_cognition_ab_held_out.v18` ;
- `paired_protocol.protocol_gate_passed=true` ;
- `terminal_mediated_exploitation_enabled_in_unified=true` ;
- `unified.controller_errors=0` ;
- `unified.actions_executed=800` ;
- `unified.experiment_actions=488` ;
- `unified.objective_distance_reductions=782` ;
- aucune discrimination SAGE.8y n'est resolue dans cet horizon : aucune
  politique SAGE.9 n'est compilee et la trajectoire reste identique a SAGE.8z ;
- `legacy_only.levels_completed=0`, `unified.levels_completed=0`,
  `legacy_only.wins=0` et `unified.wins=0`.

Validation synthetique en ligne :

- une propriete requise reste une garde et une propriete eliminee disparait de
  la correspondance ;
- une action au mauvais mode, au mauvais transfert ou au mauvais porteur est
  bloquee ;
- une action concrete ne peut pas etre repetee dans la meme branche ;
- un progres local ne promeut jamais la politique comme terminale ;
- seul un evenement terminal reel la promeut ;
- deux branches independantes sans progres la refutent ;
- la restauration observee atteint le mode de la politique ;
- l'option causale choisit la politique avant la recherche generique ;
- l'ablation ne compile ni ne selectionne aucune politique.

Validation :

- `new_sage9_tests=11 passed` ;
- `targeted_cognitive_tests=77 passed` ;
- `scoped_ruff_and_compileall=passed` ;
- `broad_repository_tests=1259 passed` sous le Python 3.11 disponible ; les
  tests historiques dependants des environnements ARC reels n'ont pas pu etre
  valides dans ce runtime, qui tente de charger des extensions `pydantic_core`
  et `scikit-learn` compilees pour Python 3.12/3.13. Les trois diagnostics reels
  SAGE.9 ont bien ete executes avec le paquet ARC 3.13 du depot precharge.

Lecture : le verrou apprentissage-vers-action est franchi. Pour la premiere
fois, une propriete de porteur apprise pendant l'examen modifie directement la
politique, restaure son contexte et produit un nouveau progres causal sans
template de solution. Ce progres ne suffit pas encore a gagner un niveau.

Le prochain verrou est la continuation multi-etape apres le premier progres de
politique. L'action SAGE.9 change le mode et la structure disponibles ; la
politique actuelle decrit seulement son etat d'entree. Il faut apprendre en
ligne une chaine de politiques sur les etats successeurs, reevaluer les
contraintes du porteur apres chaque progres et poursuivre jusqu'a un evenement
terminal, toujours sans transformer le progres local en preuve de but.

## SAGE.9a - online successor-policy chaining

Objectif :

- Capturer apres chaque action de politique le mode latent courant, toutes les
  distances terminales mesurables et un inventaire structurel sans identite
  d'objet ni template de niveau.
- Apres un progres, creer un etat successeur persistant et y reevaluer toutes
  les interventions avec les modeles directionnels appris pendant l'episode.
- Compiler une action seulement si une preuve progressive existe dans le mode
  exact. Si aucune action productive n'est connue, autoriser le noyau SAGE.9b :
  un contraste actif borne, puis apprendre son effet reel.
- Autoriser un changement d'objectif terminal dans un successeur uniquement a
  partir des hypotheses et effets deja appris en ligne.
- Reevaluer apres chaque action, limiter profondeur et essais, bloquer tout
  retour a une signature deja presente dans le chemin et restaurer seulement
  le dernier etat actif, jamais une racine depassee.
- Continuer a reserver le support terminal a un changement de niveau ou WIN
  observe.

Representation et controle :

- `SuccessorPolicyState` stocke objectif actif, mode, structures restantes,
  distances objectives, transition d'entree, action compilee ou exploratoire,
  essais par branche, enfants et statut epistemique.
- La signature d'etat combine objectif, mode, distances et structures. Les
  chemins par branche rendent la detection de cycle explicite.
- Une action progressive exacte domine un contraste actif. Un contraste avec
  gain transfere positif domine une action totalement inconnue.
- Une transition non progressive qui change la scene devient elle aussi un
  nouvel etat reevalue ; un no-op consomme le budget du meme etat.
- Le benchmark v19 expose l'ablation isolee
  `--disable-successor-policy-chaining` et toutes les metriques de capture,
  compilation, exploration, progres, profondeur, cycle et restauration
  obsolete.

Audit actif `cn04-65d47d14`, seed 0, 10 resets x 40 :

- 1 politique racine est compilee et produit 8 progres au total ;
- 4 etats issus d'un progres sont captures ;
- 3 actions successeurs sont compilees depuis une preuve exacte en ligne ;
- 3 progres supplementaires sont attribues aux etats successeurs ;
- profondeur maximale 6, `cycle_blocks=0` et
  `obsolete_restoration_blocks=0` ;
- `objective_distance_reductions=396`, contre 393 dans l'ablation SAGE.9 ;
- `experiment_actions=157`, contre 143 dans l'ablation : le gain objectif est
  obtenu au prix explicite de 14 contrastes supplementaires ;
- aucun niveau ni WIN n'est obtenu et aucun support terminal n'est invente.

Held-out long, 5 jeux public-unseen, seeds 0/1, 10 resets x 40 :

- `schema_version=sage.unified_cognition_ab_held_out.v19` et protocole valide ;
- 3929 actions reelles executees dans chaque bras ;
- 4584 reductions objectives avec la chaine, contre 4581 sans la chaine ;
- 3 progres successeurs, 3 actions compilees, profondeur 6, zero cycle ;
- 2 changements de niveau sur `ft09` dans les deux bras, donc ils constituent
  un progres ARC reel du controleur global mais ne sont pas attribuables a
  SAGE.9a ; aucun WIN.

Runtime et validation :

- la validation globale utilise desormais le runtime Python 3.12 fonctionnel
  de `ARC-AGI-3-Agents/.venv`, avec `arc_agi` et `arcengine` coherents ;
- `tests/__init__.py` empeche un paquet tiers homonyme de masquer les helpers
  locaux tout en preservant les imports historiques ;
- `new_sage9a_tests=14 passed` ;
- `benchmark_and_sage9a_tests=30 passed` ;
- `full_repository_tests=1470 passed` en 156.43 s sur l'etat final ;
- Ruff cible : passe.

Lecture : le verrou de continuation est franchi sur l'horizon long. SAGE sait
maintenant transformer un premier progres en plusieurs progres consecutifs
appris et recompiles dans leurs etats successeurs, avec un gain A/B positif et
des garde-fous effectifs. Le prochain verrou est SAGE.9c : generaliser ces
transitions entre etats structurellement analogues afin d'obtenir plus tot les
chaines utiles et de convertir ce progres local en niveaux gagnables.

## SAGE.9c - online structural successor transfer

Objectif :

- Generaliser une intervention productive vers un autre etat successeur sans
  conserver de couleur, identifiant d'objet, hash de forme ou reponse de niveau.
- Construire la generalisation exclusivement depuis un progres observe pendant
  le meme examen et ne jamais lui attribuer de valeur terminale locale.
- Conserver les contraintes qui rendent l'analogie falsifiable : famille et
  relation d'objectif, famille de mode, forme de grille, classe de distance,
  topologie multiensemble de la scene et role semantique de l'action.
- Donner priorite a une preuve exacte (`rank=58`), puis au transfert structurel
  (`rank=57`), puis au contraste actif (`rank=55/53`).
- Revoquer une classe a support unique des sa premiere contradiction et
  conserver les budgets d'essais, la detection de cycle et les restaurations
  sur l'etat successeur actif.

Representation et controle :

- `StructuralSuccessorPolicy` accumule supports, gains, branches, contextes,
  etats sources, contradictions et echecs dangereux d'un couple
  `(analogie, schema objectif, role action)` appris en ligne.
- Les signatures d'analogie retirent les couleurs et hashes mais gardent les
  classes de surface, formes topologiques, multiplicites et presence des roles
  source/cible. Elles ne contiennent ni jeu, ni niveau, ni trace de solution.
- Le compilateur peut examiner un candidat analogique meme lorsqu'une autre
  action possede deja une preuve exacte ; le classement final maintient la
  priorite de la preuve exacte.
- Le budget historique par signature de branche ne bloque plus une action
  admise dans un nouvel etat. Le budget propre a l'etat successeur et son chemin
  anti-cycle deviennent autoritaires apres un changement structurel.
- Le benchmark v20 expose l'ablation isolee
  `--disable-successor-structural-transfer` et les metriques de classes,
  predictions, selections, progres, contradictions et blocages.

Audit `cn04-65d47d14`, seed 0, 10 resets x 40 :

- 4 classes structurelles progressives sont apprises et 43 predictions de
  transfert sont admissibles ; une preuve exacte de rang superieur gagne dans
  chaque decision, donc `structural_transfer_selections=0` dans ce protocole ;
- le correctif de budget par etat porte les progres successeurs de 3 a 11,
  avec profondeur 6, zero cycle et zero restauration obsolete ; ce gain est
  present aussi dans l'ablation SAGE.9c et n'est donc pas attribue au transfert ;
- les deux bras courants sont identiques : 400 actions, 391 reductions,
  155 experiences, 0 niveau et 0 WIN ;
- aucun support terminal n'est deduit des 11 progres locaux.

Held-out long, 5 jeux public-unseen, seeds 0/1, 10 resets x 80 :

- 6837 actions reelles, 2 niveaux, 6973 reductions et 859 experiences dans
  chacun des deux bras ; aucun WIN ;
- les deux niveaux `ft09` sont donc un resultat du controleur global et non de
  SAGE.9c ;
- aucune chaine successeur n'est compilee avec ce budget de 80, alors que
  `cn04` en compile a budget 40. L'ablation est strictement neutre et aucun
  cycle n'apparait.

Validation :

- tests unitaires SAGE.9c : abstraction sans couleur/hash, transfert en ligne,
  ablation, revocation sur contradiction et budget par etat successeur ;
- `targeted_sage9c_tests=44 passed`, Ruff cible passe et
  `full_repository_tests=1475 passed` en 163.04 s sous Python 3.12 ;
- diagnostics : `sage9c_cn04_structural_transfer*.json` et
  `sage9c_held_out_long_structural_transfer*.json` ;
- le resultat ne revendique ni niveau gagne par SAGE.9c ni amelioration A/B
  nette : l'infrastructure est active et falsifiable, mais redondante avec les
  preuves exactes sur les trajectoires observees.

Lecture : SAGE.9c franchit le verrou de representation et de routage du
transfert structurel, pas encore celui de son utilite comportementale. Le
prochain verrou est SAGE.9d : rendre l'activation et la compilation des chaines
stables lorsque l'horizon change, puis mesurer un transfert selectionne et
progressif sur des etats analogues held-out avant de chercher un nouveau niveau.

## SAGE.9d - horizon-stable online learning epochs

Objectif :

- Empecher un plan operateur valide mais epistemiquement sature de monopoliser
  les actions supplementaires d'un horizon long.
- Preserver exactement la trajectoire d'apprentissage initiale : les 40
  premieres actions de chaque branche forment une epoque chaude sans nouveau
  blocage.
- Dans l'epoque longue, limiter a 12 les actions de plan operateur tant qu'aucun
  objectif deja soutenu par un resultat terminal ne progresse.
- Rearmer uniquement sur une nouvelle branche reelle ou sur le progres d'un
  objectif `terminal_supported`. Un progres d'objectif candidat ne produit ni
  valeur terminale ni budget supplementaire.
- Laisser les priorites causales, temporelles, successeures et les contrastes
  existants utiliser l'horizon libere sans creer de pseudo-branche.

Representation et audit :

- Le controleur suit le pas courant de la branche, les actions operateurs
  depuis le dernier progres terminal soutenu, le pic, les blocages et les
  rearmements.
- Le benchmark v21 expose ces metriques et l'ablation isolee
  `--disable-horizon-stable-learning-epochs`.
- Les tests verifient le blocage borne, le reset de branche et le rearmement par
  un objectif possedant deux contextes terminaux independants.
- Le seuil 12 a ete compare a 16 et 24 sur le diagnostic de developpement : les
  deux seuils plus hauts retombent a 3 ouvertures causales et ne compilent plus
  de politique a horizon 80.

Audit `cn04-65d47d14`, seed 0, 10 resets :

- A horizon 40, activation et ablation sont strictement identiques : 400
  actions, 391 reductions objectives, 155 experiences, 127 plans operateurs,
  11 progres successeurs, profondeur 6 et aucun blocage SAGE.9d.
- A horizon 80, l'activation produit 10 ouvertures causales contre 3, 10 modeles
  d'effet contre 4, 4 demandes de discrimination contre 0, 2 politiques
  compilees contre 0 et 2 progres successeurs contre 0.
- Le pic operateur passe de 67 a 25 et 360 decisions operateurs saturees sont
  bloquees ; profondeur 5, zero cycle et zero restauration obsolete.
- Aucun niveau ni WIN n'est obtenu dans ce diagnostic.

Held-out long final, 5 jeux public-unseen, seeds 0/1, 10 resets x 80 :

- `schema_version=sage.unified_cognition_ab_held_out.v21`, protocole apparie
  valide et `controller_errors=0`.
- 19 ouvertures causales contre 6 dans l'ablation, 79 observations d'effet
  medie contre 18 et 2 abstractions soutenues contre 0.
- 6 demandes de discrimination et 4 selections contre 0 ; 3 politiques sont
  compilees contre 0.
- 43 actions successeurs produisent 7 progres attribuables contre 0, avec une
  profondeur maximale 6, zero cycle et zero restauration obsolete.
- Le controleur unifie obtient 2 niveaux contre 1 pour le legacy seul. Les 2
  niveaux sont egalement presents dans l'ablation SAGE.9d : ils valident le
  controleur global, mais ne sont pas attribuables a cette iteration.
- L'activation depense 997 actions d'experience contre 859 et produit 6530
  reductions objectives contre 6973. Le gain epistemique est donc reel et
  causalement attribuable, mais pas encore converti en amelioration terminale
  nette ; aucun WIN.

Validation :

- diagnostics : `sage9d_cn04_h40*.json`, `sage9d_cn04_h80*.json` et
  `sage9d_held_out_long*.json` ;
- tests cibles controleur et benchmark : 25 passes ;
- suite complete Python 3.12 : 1477 tests passes en 193.45 s ;
- Ruff cible : passe.

Lecture : le verrou de disparition des chaines lorsque l'horizon augmente est
franchi. SAGE conserve son comportement court puis utilise l'horizon additionnel
pour rouvrir l'apprentissage causal et produire plusieurs progres successeurs.
Le prochain verrou est SAGE.9e : remplacer l'allocation fixe de la seconde
epoque par une allocation en ligne fondee sur l'incertitude causale et la
proximite d'un test terminal, afin de convertir ces chaines en changement de
niveau plutot qu'en davantage de progres locaux.

## SAGE.9e - online epistemic horizon arbitration

Objectif :

- Remplacer la reservation uniforme de SAGE.9d par une decision en ligne prise
  seulement lorsqu'un plan operateur applicable pourrait consommer l'action.
- Rendre le budget aux plans operateurs lorsqu'aucune occasion d'apprentissage
  causal ou terminal n'est observable.
- Conserver la reservation des qu'une arête causale a produit un progres ou un
  support, avant meme que l'option correspondante soit compilee.
- Renforcer la reservation en presence d'une option ouverte non resolue, d'une
  requete de replication/discrimination active, d'une abstraction mediee
  soutenue, d'une politique compilee ou d'un etat successeur encore ouvert.
- Donner la priorite maximale a un objectif proche deja `needs_contrast` ou
  `terminal_supported`. Un objectif candidat et un progres local ne suffisent
  toujours pas a declencher cette priorite terminale.

Representation et controle :

- `HorizonLearningSignals` est un paquet sans identite de jeu ni template de
  niveau : option active, arêtes productives, options ouvertes non resolues,
  requetes actives, politiques et etats successeurs, hyperarêtes soutenues,
  statut et distance du test terminal le plus proche.
- `OnlineHorizonLearningArbiter` calcule une priorite auditable et renvoie soit
  une liberation, soit un budget operateur de 12 actions ; ce budget passe a 8
  pour une demande active et a 6 pres d'un test terminal recevable.
- L'arbitre ne choisit aucune action, ne revise aucun but et n'accorde aucun
  credit. Il laisse le chemin cognitif existant utiliser les actions reservees.
- Le benchmark v22 expose l'ablation isolee
  `--disable-online-horizon-learning-arbiter`, les evaluations, reservations,
  liberations, motifs causaux/terminaux et le pic de priorite.

Audits cibles, seed 0, 10 resets x 80 :

- `tn36` ne produit aucun progres causal, option, modele medie, politique ou
  etat successeur. SAGE.9e libere donc 210 decisions : 588 plans operateurs et
  zero blocage, contre 378 plans et 210 blocages dans l'ablation SAGE.9e.
  Reductions objectives (1230), experiences (22), niveaux et WIN restent
  identiques : le budget inutile est retire sans modifier le resultat.
- `cn04` possede une arête productive des la premiere epoque. L'arbitre produit
  356 reservations causales et zero liberation, avec 201 allocations a 12 et
  155 a 8. Activation et ablation conservent exactement 264 reductions, 196
  experiences, 10 ouvertures, 2 politiques, 2 progres successeurs, profondeur
  5, zero cycle et zero restauration obsolete.

Held-out long final, 5 jeux public-unseen, seeds 0/1, 10 resets x 80 :

- `schema_version=sage.unified_cognition_ab_held_out.v22`, protocole apparie
  valide et `controller_errors=0`.
- 1147 allocations sont reellement evaluables : 727 reservations toutes
  justifiees par incertitude causale et 420 liberations toutes localisees sur
  `tn36`. Aucune allocation n'est fabriquee sur les jeux sans plan applicable.
- Les plans operateurs passent de 1174 a 1594 et les blocages de 1123 a 703.
- Toutes les metriques cognitives de resultat restent egales a l'ablation :
  6530 reductions, 997 experiences, 19 ouvertures, 79 observations mediees,
  3 politiques, 43 actions successeurs, 7 progres successeurs, profondeur 6,
  zero cycle et zero restauration obsolete.
- Le controleur conserve 2 niveaux contre 1 pour le legacy seul, mais obtient
  les memes 2 niveaux dans l'ablation SAGE.9e ; aucun WIN et aucun gain terminal
  n'est attribuable a cette iteration.
- `terminal_test_reservations=0` : aucun objectif n'atteint encore le statut
  terminal requis pour activer cette voie. Le resultat revendique une meilleure
  allocation sans regression, pas une amelioration ARC terminale.

Validation :

- 4 tests propres a l'arbitre et au routage dans le controleur ;
- tests cibles arbitre, controleur et benchmark : 30 passes ;
- suite complete Python 3.12 : 1482 tests passes en 159.59 s ;
- Ruff et compilation cibles : passes ;
- diagnostics : `sage9e_tn36_online_arbiter*.json`,
  `sage9e_cn04_online_arbiter*.json` et
  `sage9e_held_out_long_online_arbiter*.json`.

Lecture : le verrou d'allocation selective est franchi. Le prochain verrou est
SAGE.9f : lorsqu'une hypothese atteint sa postcondition sans terminer le niveau,
capturer cet etat comme frontiere terminale negative et explorer un suffixe
borne qui distingue les continuations menant a un vrai changement de niveau.
Seul le changement de niveau ou WIN devra crediter cette nouvelle continuation.

## SAGE.9f - terminal-negative frontier continuation

Objectif :

- Transformer une postcondition locale observee sans changement de niveau en
  frontiere terminale negative, jamais en preuve positive du but.
- Evaluer depuis cette frontiere un suffixe court et borne, puis separer les
  continuations par leur resultat terminal reel.
- Ne memoriser comme reussie et ne rejouer qu'une continuation ayant produit
  `level_complete`, une hausse de `levels_completed` ou `WIN`.

Representation et controle :

- `OnlineTerminalFrontierExplorer` indexe une frontiere par la signature SHA-1
  exacte de l'etat, le niveau courant et les identifiants des hypotheses dont
  la postcondition vient d'etre observee.
- Une completion explicitement predite ou selectionnee est admissible. A
  defaut, une seule hypothese completee soutenue par l'action reellement
  executee est retenue ; les autres candidats mesurables ne declenchent pas
  d'exploration speculative.
- Un seul essai peut etre ouvert par branche, avec au plus 4 essais par
  frontiere et 6 actions par suffixe. Un reset censure l'essai sans lui donner
  de credit ; `GAME_OVER` le classe dangereux ; l'expiration non terminale le
  classe negatif.
- Tant qu'aucun suffixe n'a de preuve terminale, SAGE.9f observe les decisions
  normales du controleur sans les remplacer. Les options causales, plans
  temporels, experiences, operateurs et replis restent donc inchanges. Seul un
  suffixe deja credite peut devenir prioritaire pour un replay exact, limite a
  une confirmation supplementaire.
- Le benchmark v23 expose l'ablation isolee
  `--disable-terminal-negative-frontier-exploration`, les captures, essais,
  actions observees, echecs non terminaux, dangers, credits de niveau/WIN,
  continuations reussies et replays.

Audit cible `cn04`, seed 0, 10 resets x 80 :

- SAGE.9f capture 7 frontieres, ouvre 10 essais et evalue 60 actions de suffixe.
  Aucun suffixe ne change de niveau dans sa fenetre : zero credit terminal,
  zero continuation promue et zero replay.
- L'observation est non perturbatrice : activation et ablation conservent 196
  experiences, 11 supports causaux, 1 option compilee, 13 actions successeurs,
  2 progres successeurs, zero niveau et zero WIN.

Held-out long final, 5 jeux public-unseen, seeds 0/1, 10 resets x 80 :

- Protocole apparie v23 valide et `controller_errors=0`.
- 36 frontieres, 48 essais et 288 actions de suffixe sont evalues ; aucun
  changement de niveau ni WIN n'arrive dans une fenetre active, donc zero
  credit, zero promotion et zero replay.
- Activation et ablation restent identiques sur les resultats existants : 997
  experiences, 21 supports causaux, 43 actions successeurs, 7 progres
  successeurs, 2 niveaux contre 1 pour le legacy seul, et zero WIN.
- Les deux niveaux restent ceux deja observes sur `ft09` et ne sont pas
  attribues a SAGE.9f. Cette iteration revendique la frontiere negative, la
  mesure de suffixe et la discipline de credit, pas un nouveau gain ARC.

Validation :

- tests unitaires du store, du credit terminal, du danger et du replay ;
- integration controleur et ablation benchmark : 33 tests cibles passes ;
- suite complete dans l'environnement ARC : 1488 tests passes en 162.36 s ;
- compilation ciblee : passee ;
- diagnostics : `sage9f_cn04_terminal_frontier*.json` et
  `sage9f_held_out_long_terminal_frontier*.json`.

Lecture : le controleur sait maintenant conserver un echec local comme point
de depart d'une recherche terminale sans confondre progression et victoire, et
il peut convertir un futur succes borne en replay. Le prochain verrou est que
les frontieres actuelles restent a plus de six actions d'un signal terminal.
SAGE.9g devra allouer adaptativement un horizon de continuation aux frontieres
repetees, sans perturber les chaines causales et sans utiliser de progres local
comme credit positif.
