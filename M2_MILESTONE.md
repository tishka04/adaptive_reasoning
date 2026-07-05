# M2 milestones - frontier-conditioned hypothesis expansion

Derniere mise a jour : 2026-06-29

Ce fichier sert de relais avec l'agent de reflexion. Il suit la construction de
M2 pas a pas, en gardant le contrat central : M2 imagine des hypotheses
falsifiables depuis les frontieres A40, mais ne confirme rien.

## Principe fixe

M2 ne choisit pas une action policy, ne confirme pas une mecanique, ne refute
pas une mecanique et ne modifie aucune memoire scientifique confirmee.

Chaque hypothese M2 garde :

- `status=UNRESOLVED`
- `support=0`
- `controlled_test_required=true`
- `truth_status=NOT_EVALUATED_BY_M2`
- `revision_performed=false`
- `wrong_confirmations=0`
- `trace_support_counted_as_proof=false`
- `prior_counted_as_proof=false`

M2 lit A40 et ecrit uniquement dans `diagnostics/m2/`.

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| M2.0 - Schema-first contract | Fait | `theory/m2/schema.py`, `theory/m2/validators.py`, `tests/test_m2_schema.py`, `tests/test_m2_validators.py` | Schema strict, hypotheses invalides rejetees |
| M2.1 - A40 frontier intake | Fait | `theory/m2/frontier_intake.py`, `tests/test_m2_frontier_intake.py`, `diagnostics/m2/frontier_intake.json` | Lit A40, filtre les frontieres pretes, accepte entree vide |
| M2.2 - Heuristic generator | Fait | `theory/m2/heuristic_generator.py`, `tests/test_m2_heuristic_generator.py` | Genere au moins 4 hypotheses bp35 |
| M2.3 - Normalizer adversarial | Fait | `theory/m2/normalizer.py`, `tests/test_m2_normalizer.py`, `tests/test_m2_adversarial_raw_proposals.py` | Corrige/rejette sorties adversariales |
| M2.4 - Metric registry and compiler | Fait | `theory/m2/metric_registry.py`, `theory/m2/testability_compiler.py`, `tests/test_m2_metric_registry.py`, `tests/test_m2_testability_compiler.py` | Compile des requests `READY_FOR_M3` |
| M2.5 - Dedup and priority | Fait | `theory/m2/hypothesis_merger.py`, `tests/test_m2_hypothesis_merger.py`, `diagnostics/m2/generation_audit.json` | Sources fusionnees, priorite jamais support |
| M2.6 - First bp35 frontier run | Fait | `diagnostics/m2/frontier_conditioned_hypotheses.json`, `diagnostics/m2/m3_candidate_experiment_requests.json` | Run heuristique sur les 2 frontieres A40 actuelles |
| M2.7 - Controlled handoff to M3 | Fait | `theory/m2/m3_handoff.py`, `tests/test_m2_m3_handoff.py`, `diagnostics/m2/m3_handoff_validation.json` | M3 peut charger et classer les requests M2 |
| M2.8 - Synthetic multi-frontier stress | Fait | `theory/m2/synthetic_frontier_stress.py`, `tests/test_m2_multi_frontier_synthetic.py` | 4 familles distinctes, invalides rejetes proprement |
| M2.9 - Real frontier stress | Fait | `theory/m2/real_frontier_stress_test.py`, `tests/test_m2_real_frontier_stress_test.py` | Ratios reportes sans minimum rigide global |
| M2.10 - LLM mock adapter | Fait | `theory/m2/llm_adapter.py`, `theory/m2/mock_llm_generator.py`, `tests/test_m2_llm_adapter.py`, `tests/test_m2_llm_adversarial_outputs.py` | Interface stable, sorties adversariales normalisees |
| M2.11 - World model mock adapter | Fait | `theory/m2/world_model_adapter.py`, `theory/m2/mock_world_model.py`, `diagnostics/m2/mock_world_model_predictions.json`, `diagnostics/m2/mock_world_model_hypotheses.json`, `tests/test_m2_world_model_adapter.py` | Scores utilises comme priorite, pas comme evidence |
| M2.12 - Multi-source merger | Fait | `tests/test_m2_multi_source_merger.py` | Fallback heuristique, support toujours zero |
| M2.12b - Context replay canonicalizer | Fait | `theory/m2/context_replay.py`, `tests/test_m2_context_replay_canonicalizer.py` | `after_ACTION3_live_after_ACTION6` rejoue `ACTION6`, puis `ACTION3` |
| M2.12c - Blocked-skill precondition guard | Fait | `theory/m2/precondition_guard.py`, `tests/test_m2_blocked_skill_precondition_guard.py` | Skill bloquee + precondition active devient `BLOCKED_NOT_TESTABLE` |
| M2.12d - True multi-source collision fixture | Fait | `theory/m2/multi_source_collision_fixture.py`, `tests/test_m2_multi_source_collision_fixture.py`, `diagnostics/m2/multi_source_collision_fixture.json` | Collision heuristic/LLM fusionnee, support reste zero |
| M2.12e - M3 contextual execution smoke | Fait | `theory/m2/m3_execution_smoke.py`, `tests/test_m2_m3_execution_smoke.py`, `diagnostics/m2/m3_execution_smoke.json` | Rejoue le contexte exact avant target vs control |
| M2.O1 - Objective-conditioned hypothesis generation | Fait | `theory/m2/objective_conditioned_hypotheses.py`, `tests/test_m2_objective_conditioned_hypotheses.py`, `diagnostics/m2/objective_conditioned_hypotheses.json` | 1 requete objectif consommee, 4 hypotheses stop/switch testables, support zero |
| M2.G1 - Objective-conversion hypothesis generation | Fait | `theory/m2/objective_conversion_hypothesis_generator.py`, `tests/test_m2_objective_conversion_hypothesis_generator.py`, `diagnostics/m2/objective_conversion_hypotheses.json` | 1 requete P2.G3 consommee, 12 hypotheses objective-conversion testables, support zero |
| M2.G2 - Risk-aware objective-completion hypothesis generation | Fait | `theory/m2/risk_aware_objective_completion_hypothesis_generator.py`, `tests/test_m2_risk_aware_objective_completion_hypothesis_generator.py`, `diagnostics/m2/risk_aware_objective_completion_hypotheses.json` | 1 requete P2.G5 consommee, 15 hypotheses readiness/commit/goal/discriminator/selector-gap testables, support zero |
| M2.13a - State-conditioned local LLM generator | Fait | `theory/m2/local_llm_generator.py`, `theory/m2/metric_registry.py`, `tests/test_m2_local_llm_generator.py`, `tests/test_m2_local_llm_adversarial_outputs.py`, `tests/test_m2_metric_registry.py`, `diagnostics/m2/state_conditioned_llm_hypotheses.json`, `diagnostics/m2/state_conditioned_llm_m3_candidate_requests.json` | LLM offline, desactive par defaut, mock state-conditioned teste; 6 hypotheses 6 familles, 5 ready_for_m3, 1 unlock bloque candidate-only; support=0 |
| M2.14a/b/c - ARC-LeWM foundation | Fait | `theory/m2/object_world_model_generator.py`, `theory/m2/arc_lewm_dataset.py`, `theory/m2/arc_lewm_model.py`, `theory/m2/arc_lewm_losses.py`, `theory/m2/arc_lewm_trainer.py`, `diagnostics/m2/object_world_model_invariant_packet.json`, `training/m2_arc_lewm_transitions.jsonl`, `diagnostics/m2/arc_lewm_dataset_manifest.json`, `diagnostics/m2/arc_lewm_source_audit.json`, `models/m2_arc_lewm.pt`, `diagnostics/m2/arc_lewm_training_report.json` | Dataset 6 jeux genere; checkpoint selectionne best-val present; no collapse; support=0 |
| M2.14c+ - Best-val/baseline trainer diagnostics | Fait | `theory/m2/arc_lewm_trainer.py`, `tests/test_m2_arc_lewm_trainer_smoke.py`, `models/m2_arc_lewm_best_val.pt`, `models/m2_arc_lewm_last.pt`, `diagnostics/m2/arc_lewm_training_report.json` | Run reel enrichi genere; best_val_epoch=2; baseline action-conditionnee positive; per-game validation presente; support=0 |
| M2.14d/e-mini - ARC-LeWM signal + hypothesis adapter | Fait | `theory/m2/arc_lewm_signal_extractor.py`, `theory/m2/arc_lewm_hypothesis_adapter.py`, `tests/test_m2_arc_lewm_signal_extractor.py`, `tests/test_m2_arc_lewm_hypothesis_adapter.py`, `diagnostics/m2/arc_lewm_signal_report.json`, `diagnostics/m2/arc_lewm_hypotheses.json`, `diagnostics/m2/arc_lewm_m3_candidate_requests_v2.json`, `diagnostics/m2/arc_lewm_m3_handoff_validation.json` | 4 hypotheses; 1 request READY_FOR_M3; 3 bloquees; aucun RESET/unknown_game ready; ACTION6 args preserves; handoff M3 valide |
| M2.15 - LLM + world-model fused generator | Fait | `theory/m2/fused_hypothesis_generator.py`, `tests/test_m2_fused_hypothesis_generator.py`, `tests/test_m2_fused_hypothesis_adversarial_outputs.py`, `diagnostics/m2/fused_llm_wm_input_packet.json`, `diagnostics/m2/fused_llm_wm_hypotheses.json`, `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`, `diagnostics/m2/fused_llm_wm_handoff_validation.json` | 4 familles fusees; 3 requests M3 offline-trace ready; 1 contrefactuel offline bloque; M3.7e execute sans nouvelle evidence independante; M3.7f confirme qu'il manque un env state restorable; support=0 |

## Commandes de verification

Tests M2 cibles :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_m1_controlled_followup_experiment.py `
  tests\test_m2_schema.py `
  tests\test_m2_validators.py `
  tests\test_m2_frontier_intake.py `
  tests\test_m2_heuristic_generator.py `
  tests\test_m2_normalizer.py `
  tests\test_m2_adversarial_raw_proposals.py `
  tests\test_m2_metric_registry.py `
  tests\test_m2_testability_compiler.py `
  tests\test_m2_hypothesis_merger.py `
  tests\test_m2_m3_handoff.py `
  tests\test_m2_multi_frontier_synthetic.py `
  tests\test_m2_real_frontier_stress_test.py `
  tests\test_m2_llm_adapter.py `
  tests\test_m2_llm_adversarial_outputs.py `
  tests\test_m2_world_model_adapter.py `
  tests\test_m2_multi_source_merger.py `
  tests\test_m2_context_replay_canonicalizer.py `
  tests\test_m2_blocked_skill_precondition_guard.py `
  tests\test_m2_multi_source_collision_fixture.py `
  tests\test_m2_m3_execution_smoke.py `
  -q
```

## M2.14 - ARC-LeWM foundation + c+/d/e-mini

Statut :

- M2.14a/b/c : fait. Le packet semantique, le dataset row-local, le modele
  latent, SIGReg et le checkpoint selectionne existent.
- M2.14c+ : fait. Le run reel enrichi a genere best-val/last, les baselines
  persistence/action-agnostic et la validation par jeu.
- M2.14d/e-mini : fait. Le signal extractor a produit un report reel et
  l'adapter a genere des hypotheses candidate-only plus un handoff M3 v2
  strictement filtre.

Ajouts :

- `theory/m2/object_world_model_generator.py` :
  packet semantique `mechanistic_context_candidates`, candidate-only.
- `theory/m2/arc_lewm_dataset.py` :
  dataset row-local `frame_before + action -> frame_after` depuis
  `human_traces/*.steps.jsonl`.
- `theory/m2/arc_lewm_model.py`, `arc_lewm_losses.py`, `arc_lewm_trainer.py` :
  encodeur latent, predicteur action-conditionne, SIGReg, detecteur collapse,
  checkpoint best-val/last, baselines persistence/action-agnostic et
  validation par jeu.
- `theory/m2/arc_lewm_signal_extractor.py` :
  M2.14d-mini, extraction de signaux latents interpretable-priority seulement.
- `theory/m2/arc_lewm_hypothesis_adapter.py` :
  M2.14e-mini, conversion des signaux WM en hypotheses M2 normalisees et
  requests M3.

Artefacts deja generes :

- `diagnostics/m2/object_world_model_invariant_packet.json`
- `training/m2_arc_lewm_transitions.jsonl`
- `diagnostics/m2/arc_lewm_dataset_manifest.json`
- `diagnostics/m2/arc_lewm_source_audit.json`
- `models/m2_arc_lewm.pt`
- `models/m2_arc_lewm_best_val.pt`
- `models/m2_arc_lewm_last.pt`
- `diagnostics/m2/arc_lewm_training_report.json`
- `diagnostics/m2/arc_lewm_signal_report.json`
- `diagnostics/m2/arc_lewm_hypotheses.json`
- `diagnostics/m2/arc_lewm_m3_candidate_requests_v2.json`
- `diagnostics/m2/arc_lewm_m3_handoff_validation.json`

Etat trainer reel :

- best_val_epoch = 2
- best_val_prediction_loss = 0.1564526769721513
- best_val_total_loss = 0.30723413832495
- action_conditioning_utility = `POSITIVE`
- beats_action_agnostic_baseline = true
- beats_persistence_baseline = false
- per_game_validation present sur `dc22-4c9bff3e` et `ft09-0d8bbf25`

Etat dataset :

- transitions = 5711
- games_total = 6
- split = 4 train / 2 val
- transition_alignment_policy =
  `row_local_frame_before_action_frame_after`
- continuity_mismatches = 0

Etat handoff d/e-mini v2 :

- `diagnostics/m2/arc_lewm_m3_candidate_requests_v2.json`
- `diagnostics/m2/arc_lewm_m3_candidate_requests.json` est un artefact legacy
  v1 et ne doit pas etre consomme par M3 pour ARC-LeWM.
- experiment_requests = 4
- ready_for_m3 = 1
- blocked_not_testable = 3
- blocked_by_reason = `BLOCKED_RESET_BOUNDARY: 2`,
  `DIAGNOSTIC_ONLY_AGGREGATE: 1`
- reset_actions_marked_ready_for_m3 = 0
- unknown_game_marked_ready_for_m3 = 0
- la request prete cible `ACTION6` avec `target_action_args={"x":45,"y":30}`,
  `source_episode_id=3c719b36eb4d`, `source_step=239`,
  `replayability=OFFLINE_TRACE_CONTEXT_ONLY`
- `diagnostics/m2/arc_lewm_m3_handoff_validation.json` :
  invalid_m3_requests = 0, M3 peut ranker et executer au moins une request

Run M3 aval M3.7c execute :

- `diagnostics/m3/arc_lewm_m2_candidate_experiment_results.json`
- request executee :
  `m2m3::m2_14_lewm::terminal_like_latent_neighborhoods::004`
- execution_mode = `offline_trace_context`
- source_transition_id =
  `m2_14d::dc22-4c9bff3e::3c719b36eb4d::0239`
- target = `ACTION6 {"x":45,"y":30}`
- control = `ACTION3`
- matched_control_samples = 195
- target terminal_rate = 1.0
- control terminal_rate = 0.0
- support_events = 1
- contradiction_events = 0
- support = 0
- A32/A33 non modifies

Run M3 aval M3.7d replication :

- `diagnostics/m3/arc_lewm_terminal_risk_replication_requests.json`
- `diagnostics/m3/arc_lewm_terminal_risk_replication_results.json`
- signaux sources :
  `terminal_like_latent_neighborhoods` depuis
  `diagnostics/m2/arc_lewm_signal_report.json`
- requests READY generees = 14
- requests executees = 14
- source_transition_id distincts = 14
- episodes distincts = 14
- jeux couverts = 4
- actions cibles couvertes = 6
- controles = `same_game_same_available_actions` pour les 14 experiences
- target_trace_samples_total = 14
- matched_control_samples_total = 3357
- support_events = 14
- independent_source_support_events = 14
- duplicate_source_support_events = 0
- unique_context_support_events = 7
- context_reused_support_events = 7
- contradiction_events = 0
- support = 0
- A32/A33 non modifies

Lecture M2.14 -> M3 :

- M2.14 produit maintenant un signal terminal-risk ARC-LeWM que M3 sait
  tester en offline-trace-context sur plusieurs sources distinctes.
- Ce resultat valide la chaine WM -> hypotheses candidates -> handoff strict
  -> execution M3 grounded.
- Il ne confirme pas encore une mecanique : les observations restent
  candidate-only et `support=0`.

Contrat conserve :

- `support=0`
- `truth_status=NOT_EVALUATED_BY_M2`
- `revision_status=CANDIDATE_ONLY`
- `world_model_score_counted_as_support=false`
- `world_model_counted_as_evidence=false`
- aucun write A32/A33

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_m2_schema.py `
  tests\test_m2_validators.py `
  tests\test_m2_testability_compiler.py `
  tests\test_m2_m3_handoff.py `
  tests\test_m3_m2_candidate_experiment_runner.py `
  tests\test_m2_object_world_model_generator.py `
  tests\test_m2_object_world_model_adversarial_inputs.py `
  tests\test_m2_arc_lewm_dataset.py `
  tests\test_m2_arc_lewm_model.py `
  tests\test_m2_arc_lewm_losses.py `
  tests\test_m2_arc_lewm_trainer_smoke.py `
  tests\test_m2_arc_lewm_signal_extractor.py `
  tests\test_m2_arc_lewm_hypothesis_adapter.py `
  tests\test_m3_arc_lewm_terminal_risk_replication.py -q
```

Dernier resultat cible M2.14 + schema/handoff + M3 offline trace/replication :
48 tests passes.

## M2.15 - LLM + world-model fused generator

Statut : fait.

Objectif :

Fusionner les hypotheses abductives LLM M2.13a, les invariants symboliques
candidate-only, les signaux ARC-LeWM M2.14 et les resultats M3.7c/M3.7d/M3.G6
en nouvelles hypotheses falsifiables. M2.15 ne choisit pas de policy, ne
confirme rien et ne compte ni LLM, ni world model, ni `support_events` M3 comme
support.

Entrees :

- `diagnostics/m2/state_conditioned_llm_hypotheses.json`
- `diagnostics/m2/state_conditioned_llm_m3_candidate_requests.json`
- `diagnostics/m2/object_world_model_invariant_packet.json`
- `diagnostics/m2/arc_lewm_signal_report.json`
- `diagnostics/m2/arc_lewm_hypotheses.json`
- `diagnostics/m2/arc_lewm_m3_candidate_requests_v2.json`
- `diagnostics/m3/arc_lewm_m2_candidate_experiment_results.json`
- `diagnostics/m3/arc_lewm_terminal_risk_replication_results.json`
- `diagnostics/m3/risk_aware_objective_completion_experiment_results.json`

Sorties :

- `diagnostics/m2/fused_llm_wm_input_packet.json`
- `diagnostics/m2/fused_llm_wm_hypotheses.json`
- `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`
- `diagnostics/m2/fused_llm_wm_handoff_validation.json`
- `diagnostics/m3/fused_llm_wm_experiment_results.json` (run M3 aval)
- `diagnostics/m3/offline_frame_counterfactual_feasibility.json`
  (probe M3 aval)

Familles bornees :

- `terminal_risk_precondition`
- `terminal_safe_alternative_action`
- `wm_llm_disagreement_frontier`
- `objective_completion_vs_terminal_risk_tradeoff`

Familles explicitement interdites :

- `general_terminal_predictor`
- `universal_action_safety_model`
- `confirmed_goal_representation`

Run du 2026-06-29 :

- input packet consomme = true
- terminal_risk_support_events_read = 14
- terminal_risk_support_events_counted_as_support = false
- hypotheses_generated = 4
- ready_for_m3_candidate_experiment_request = 3
- blocked_not_testable_hypotheses = 1
- blocking reason =
  `BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME`
- M3 requests = 4
- ready_for_m3 = 3
- offline_trace_context_requests = 3
- invalid_m3_requests = 0
- M3 selector can rank = true
- M3 can execute at least one request = true
- `context_replay=[]` implique `context_replay_args=null`
- support = 0
- truth_status = `NOT_EVALUATED_BY_M2`
- revision_status = `CANDIDATE_ONLY`
- llm_output_counted_as_evidence = false
- world_model_score_counted_as_support = false
- world_model_counted_as_evidence = false
- m3_observation_counted_as_confirmation = false
- policy_generated = false
- a32_write_performed = false
- a33_write_performed = false

Hypotheses produites :

- `m2_15_fused::terminal_risk_precondition::001`
- `m2_15_fused::terminal_safe_alternative_action::002`
- `m2_15_fused::wm_llm_disagreement_frontier::003`
- `m2_15_fused::objective_completion_vs_terminal_risk_tradeoff::004`

Run M3 aval M3.7e :

- entree :
  `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`
- sortie :
  `diagnostics/m3/fused_llm_wm_experiment_results.json`
- requests executees = 3
- request skippee =
  `m2m3::m2_15_fused::terminal_safe_alternative_action::002`
- raison du skip =
  `BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME`
- support_events = 3
- contradiction_events = 0
- source_transition_reused_from_m3_7d = true
- reused_source_support_events = 3
- independent_source_support_events = 0
- new_independent_terminal_risk_evidence = false
- fusion_hypothesis_routing_validated = true
- matched_control_samples_total = 784
- target_trace_samples_total = 3
- support = 0
- A32/A33 non modifies

Run M3 aval M3.7f :

- entree :
  `diagnostics/m2/fused_llm_wm_m3_candidate_requests.json`
- sortie :
  `diagnostics/m3/offline_frame_counterfactual_feasibility.json`
- request inspectee =
  `m2m3::m2_15_fused::terminal_safe_alternative_action::002`
- source_transition_id =
  `m2_14d::ar25-e3c63847::be48a96d70f0::0295`
- target alternative = `ACTION3`
- target_action_available_in_frame = true
- frame_before_grid_present = true
- full_env_state_payloads_present = 0
- restore_contracts_detected = 0
- can_reconstruct_env_state = false
- can_replay_alternative_action = false
- replay_exact_hashable = false
- counterfactual_feasibility =
  `BLOCKED_REQUIRES_ENV_STATE_RESTORE_OR_ACTIVE_COLLECTION`
- recommended_frontier_type =
  `NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT`
- frontier_write_performed = false
- support = 0
- A32/A33 non modifies

Lecture :

M2.15 transforme la chaine M2.13a + M2.14 + M3.7d en quatre questions
experimentales nouvelles. Les trois hypotheses pretes peuvent etre rejouees
depuis `OFFLINE_TRACE_CONTEXT_ONLY`. L'hypothese
`terminal_safe_alternative_action` est volontairement bloquee, car elle demande
un contrefactuel depuis une frame offline : l'executor M3 courant mesure la
transition observee, pas encore l'action alternative sur le meme frame-before.
Le run M3 aval valide le routage fused LLM+WM, mais ne cree aucune evidence
terminale independante nouvelle, car les trois sources etaient deja dans M3.7d.
Le probe M3.7f confirme que cette limite est experimentale : `grid_t` et les
actions disponibles existent, mais aucun etat env complet/restorable n'est
present dans la trace offline.

Commande :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.fused_hypothesis_generator `
  --packet-out diagnostics\m2\fused_llm_wm_input_packet.json `
  --out diagnostics\m2\fused_llm_wm_hypotheses.json `
  --m3-out diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --handoff-out diagnostics\m2\fused_llm_wm_handoff_validation.json
```

Run M3 fused :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.fused_llm_wm_experiment_runner `
  --m2-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --m3-7d-results diagnostics\m3\arc_lewm_terminal_risk_replication_results.json `
  --offline-trace-dataset training\m2_arc_lewm_transitions.jsonl `
  --out diagnostics\m3\fused_llm_wm_experiment_results.json
```

Probe contrefactuel M3.7f :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m3.offline_frame_counterfactual_probe `
  --m2-requests diagnostics\m2\fused_llm_wm_m3_candidate_requests.json `
  --offline-trace-dataset training\m2_arc_lewm_transitions.jsonl `
  --out diagnostics\m3\offline_frame_counterfactual_feasibility.json
```

Verification :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest `
  tests\test_m1_controlled_followup_experiment.py `
  tests\test_m2_schema.py `
  tests\test_m2_validators.py `
  tests\test_m2_testability_compiler.py `
  tests\test_m2_m3_handoff.py `
  tests\test_m3_m2_candidate_experiment_runner.py `
  tests\test_m2_local_llm_generator.py `
  tests\test_m2_local_llm_adversarial_outputs.py `
  tests\test_m2_arc_lewm_signal_extractor.py `
  tests\test_m2_arc_lewm_hypothesis_adapter.py `
  tests\test_m3_arc_lewm_terminal_risk_replication.py `
  tests\test_m2_fused_hypothesis_generator.py `
  tests\test_m2_fused_hypothesis_adversarial_outputs.py `
  tests\test_m3_fused_llm_wm_experiment_runner.py `
  tests\test_m3_offline_frame_counterfactual_probe.py -q
```

Dernier resultat cible M2.15 + LLM/M2.14/M3.7d/M3.7e/M3.7f guards :
65 tests passes.

Run A40 -> M2 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.frontier_conditioned_hypotheses `
  --frontiers diagnostics\a40\frontier_handoff_requests.json `
  --out diagnostics\m2\frontier_conditioned_hypotheses.json `
  --m3-out diagnostics\m2\m3_candidate_experiment_requests.json
```

Validation handoff M3 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.m3_handoff `
  --m3-requests diagnostics\m2\m3_candidate_experiment_requests.json `
  --out diagnostics\m2\m3_handoff_validation.json
```

Guard anti-confirmation :

```powershell
$patterns = @(
  'truth_status": "' + 'CONFIRMED',
  'status": "' + 'CONFIRMED',
  'revision_performed": ' + 'true',
  'wrong_confirmations": ' + '1',
  'support": ' + '1'
)

Select-String -Path theory\m2\*.py,diagnostics\m2\*.json,M2_MILESTONE.md -Pattern $patterns -SimpleMatch
```

## Lecture

M2 transforme les frontieres A40 en questions experimentales. Les heuristiques,
le mock LLM et le mock world model peuvent proposer des idees, mais toutes les
sorties passent par le meme normalizer et le meme validator. Les scores de
priorite servent seulement a ordonner la file pour M3 ; ils ne deviennent jamais
du support, de l'evidence ou une confirmation.

## Run de consolidation M2.12b-e - 2026-06-19

Corrections appliquees :

- `after_ACTION3_live_after_ACTION6` est canonise en replay
  `["ACTION6", "ACTION3"]`.
- Les hypotheses `candidate_action == blocked_skill` avec precondition encore
  active deviennent non testables si elles n'incluent pas explicitement un
  mecanisme de reset/precondition.
- Une collision volontaire heuristic/LLM est fusionnee en une seule hypothese
  avec `sources=["heuristic", "llm"]`.
- Un smoke M3 contextuel rejoue `ACTION6 -> ACTION3`, execute `ACTION4` contre
  un controle dynamique, et ecrit uniquement des evenements experimentaux.

KPIs du run heuristique :

- `frontier_requests_consumed=2`
- `hypotheses_generated=4`
- `experiment_requests_ready_for_m3=4`
- `final_invalid_hypotheses=0`
- `wrong_confirmations=0`

KPIs mock world model apres garde precondition :

- `hypotheses_generated=10`
- `testable_hypotheses=9`
- `blocked_not_testable_hypotheses=1`
- `experiment_requests_ready_for_m3=9`
- `wrong_confirmations=0`

Smoke M3 contextuel :

- `m3_context_replay_exact=true`
- replay concret retrouve : `ACTION6 {"x":18,"y":0}` puis `ACTION3 {}`
- `target_action=ACTION4`
- `control_action=ACTION3`
- `controlled_experiments_run=1`
- `support_events=0`
- `contradiction_events=0`
- `support=0`
- `revision_performed=false`
- `wrong_confirmations=0`

Lecture :

- Le contexte causal est maintenant correct.
- L'experience `ACTION4` apres saturation est executable, mais neutre sous la
  metrique `local_patch_before_after` dans ce smoke.
- Cette neutralite n'est pas une refutation M2 ; elle devient une observation
  experimentale candidate que M3/A32 peuvent exploiter ensuite.

## M2.O1 - Objective-conditioned hypothesis generation

Objectif :

Transformer la requete objectif P2.6 en hypotheses candidates falsifiables sur
le probleme stop/switch. M2.O1 ne confirme rien et ne cree pas encore les tests
M3 ; il prepare seulement des hypotheses `ready_for_m3_candidate_experiment_request`.

Entree :

- `diagnostics/p2/bp35_objective_frontier_handoff_requests.json`

Sortie :

- `diagnostics/m2/objective_conditioned_hypotheses.json`

Contrat :

- Lire uniquement la requete objectif P2.6.
- Refuser une source `support>0`, `a33_ready=true`, ou prete pour handoff saturation.
- Generer des hypotheses autour de :
  - `stop_switch_criterion`
  - `terminal_risk_predictor`
  - `subgoal_switch_after_local_affordance`
  - `global_objective_alignment_metric`
- Chaque hypothese reste `UNRESOLVED`, `support=0`, `truth_status=NOT_EVALUATED_BY_M2`.
- `revision_status=CANDIDATE_ONLY` est conserve dans l'artefact objectif.
- Aucun write M3/A32/A33.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.objective_conditioned_hypotheses `
  --objective-frontier-requests diagnostics\p2\bp35_objective_frontier_handoff_requests.json `
  --out diagnostics\m2\objective_conditioned_hypotheses.json
```

Resume du run :

- objective_requests_seen = 1
- objective_requests_consumed = 1
- objective_requests_rejected = 0
- hypothesis_batches = 1
- hypotheses_generated = 4
- testable_hypotheses = 4
- ready_for_m3_candidate_experiment_request = 4
- blocked_not_testable_hypotheses = 0
- final_invalid_hypotheses = 0
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_M2`
- wrong_confirmations = 0

Hypotheses produites :

- `stop_switch_criterion` : tester si continuer `ACTION6` augmente le risque terminal par rapport a stop/switch.
- `terminal_risk_predictor` : tester si un long prefixe `ACTION6` augmente le taux `GAME_OVER`.
- `subgoal_switch_after_local_affordance` : tester si switcher vers un autre sous-but ameliore les niveaux completes.
- `global_objective_alignment_metric` : tester si une metrique objectif predit mieux l'echec terminal que les effets locaux.

Lecture :

M2.O1 transforme la frontier objectif P2.6 en hypotheses falsifiables. Il ne propose pas une policy, ne confirme pas la regle patch-similar et ne compte pas le resultat P1/P2 comme preuve. La suite naturelle est `M3.O1`, qui doit compiler ces hypotheses en tests controles : continuer `ACTION6` vs stopper/switcher, varier la longueur de prefixe et comparer metriques locales contre metriques objectif.

## M2.G1 - Objective-conversion hypothesis generation

Objectif :

Transformer la requete P2.G3 `OBJECTIVE_CONVERSION_FRONTIER_REQUEST` en
hypotheses falsifiables sur la conversion objectif apres safe-stop. M2.G1 ne
fait aucune execution, ne planifie pas de rollout policy et ne confirme rien.

Entree :

- `diagnostics/p2/objective_conversion_handoff_requests.json`

Sortie :

- `diagnostics/m2/objective_conversion_hypotheses.json`

Contrat :

- Lire uniquement la requete objective-conversion P2.G3.
- Refuser une source `support>0`, `a33_ready=true`, `ready_for_direct_downstream_write=true`, ou avec write aval deja effectue.
- Generer des hypotheses autour de :
  - `post_safe_stop_objective_conversion`
  - `subgoal_target_reselection`
  - `objective_readiness_condition`
  - `terminal_safe_sequence_search`
- Chaque hypothese garde `status=UNRESOLVED`, `support=0`, `truth_status=NOT_EVALUATED_BY_M2`.
- Chaque hypothese porte un `falsification_signal`.
- Chaque hypothese est rattachee a un `requested_experiment_style` emis par P2.G3.
- Aucun write M3/A32/A33.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.objective_conversion_hypothesis_generator `
  --objective-conversion-requests diagnostics\p2\objective_conversion_handoff_requests.json `
  --out diagnostics\m2\objective_conversion_hypotheses.json
```

Resume du run :

- objective_conversion_requests_seen = 1
- objective_conversion_requests_consumed = 1
- objective_conversion_requests_rejected = 0
- hypothesis_batches = 1
- hypotheses_generated = 12
- testable_hypotheses = 12
- ready_for_m3_candidate_experiment_request = 12
- blocked_not_testable_hypotheses = 0
- final_invalid_hypotheses = 0
- all_requested_hypothesis_families_covered = true
- all_hypotheses_have_falsification_signal = true
- all_hypotheses_map_to_requested_experiment_style = true
- execution_performed = false
- policy_rollout_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_M2`
- wrong_confirmations = 0

Hypotheses produites :

- `post_safe_stop_objective_conversion`
  - `ACTION3` apres safe stop peut convertir le progres relationnel en progres objectif.
  - `ACTION4` apres safe stop peut convertir le progres relationnel en progres objectif.
  - `ACTION6` peut re-activer une conversion objectif seulement dans certains contextes de readiness.
- `subgoal_target_reselection`
  - `distance_decreases` vers E136/E137/E138/E139 n'est peut-etre plus la bonne cible apres safe stop.
  - une relation-cible non la plus proche peut mieux predire la completion.
  - une transition de configuration globale peut etre un meilleur signal qu'une relation locale.
- `objective_readiness_condition`
  - une condition de readiness peut etre requise avant l'action de conversion.
  - la readiness peut dependre d'un plateau relationnel.
  - la readiness peut dependre du HUD/horizon ou du budget d'actions consommees.
- `terminal_safe_sequence_search`
  - `ACTION3,ACTION4` peut convertir mieux qu'une action simple.
  - `ACTION4,ACTION3` peut convertir mieux qu'une action simple.
  - `ACTION6,ACTION3` ou `ACTION6,ACTION4` peut convertir seulement avec terminal guard.

Styles d'experience couverts :

- `stop_state_action_matrix`
- `post_safe_stop_short_sequence_probe`
- `relation_target_ablation_after_safe_stop`
- `objective_completion_vs_relation_progress_discriminator`

Lecture :

M2.G1 transforme la frontier safe-but-passive P2.G3 en questions scientifiques
testables. Il ne choisit pas de policy de conversion, ne teste aucune sequence
et ne compte pas P3.G1 comme preuve. La suite naturelle est `M3.G1`, qui devra
compiler ces hypotheses en experiences controlees depuis des etats safe-stop :
actions simples, sequences courtes, ablations de relation-cible et discriminateur
completion objectif vs progres relationnel.

Suite (faite) : `M3.G1` est implemente (planner G1.1, executor G1.2,
consolidation G1.3 ; voir `M3_MILESTONES.md`). Les 12 hypotheses ont ete
compilees en 12 requests candidate-only, executees depuis un safe-stop P3.G1
rejoue, et consolidees en `MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY` : signal de
conversion candidate-only pour `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4`
(best `ACTION6,ACTION3`, delta_vs_hold +15, sans terminal re-entry), alors que
`ACTION3`/`ACTION4` et la continuation relation-progress re-entrent en terminal.
Toujours `support=0`, `truth_status=NOT_EVALUATED_BY_M3`, aucun verdict.

## M2.G2 - Risk-aware objective-completion hypothesis generation

Objectif :

Transformer la requete P2.G5 `RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST`
en hypotheses falsifiables sur ce qui manque apres une conversion post-stop
safe/risk-aware. M2.G2 ne reteste pas simplement `ACTION6,ACTION3` ou
`ACTION6,ACTION4` : ces sequences sont le substrat policy deja observe. Le
generateur cible readiness, commit action, representation du but,
discriminateur proxy-vs-completion et gap du selector.

Entree :

- `diagnostics/p2/risk_aware_objective_frontier_handoff_requests.json`

Sortie :

- `diagnostics/m2/risk_aware_objective_completion_hypotheses.json`

Contrat :

- Lire uniquement la requete P2.G5 comme entree obligatoire.
- Ne pas lire A33, LLM, world model, et ne pas executer de rollout.
- Refuser une source `support>0`, `a33_ready=true`, `ready_for_direct_downstream_write=true`, ou avec write aval deja effectue.
- Accepter uniquement :
  - `handoff_type=RISK_AWARE_OBJECTIVE_COMPLETION_FRONTIER_REQUEST`
  - `target=M2_OR_M3`
  - `target_modules=["M2.G2"]`
  - `frontier_type=RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER`
  - `blocked_capability=objective_completion_after_risk_aware_safe_conversion`
- Generer des hypotheses autour de :
  - `objective_readiness_detection`
  - `post_conversion_commit_action_search`
  - `goal_state_representation_beyond_safe_progress`
  - `proxy_progress_vs_completion_discriminator`
  - `risk_aware_selector_completion_gap`
- Ne generer aucune hypothese dont la cible est simplement `ACTION6`, `ACTION6,ACTION3` ou `ACTION6,ACTION4`.
- Chaque hypothese garde `status=UNRESOLVED`, `support=0`, `truth_status=NOT_EVALUATED_BY_M2`.
- Chaque hypothese porte `required_observables`, `forbidden_interpretations`, `falsification_signal`, `ready_for_m3_g5=true`, `ready_for_a32=false`, `ready_for_a33=false`.
- `execution_performed=false`, `policy_rollout_performed=false`, `environment_step_performed=false`.
- Aucun write M3/A32/A33.

Run :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.risk_aware_objective_completion_hypothesis_generator `
  --risk-aware-objective-requests diagnostics\p2\risk_aware_objective_frontier_handoff_requests.json `
  --out diagnostics\m2\risk_aware_objective_completion_hypotheses.json
```

Resume du run :

- risk_aware_objective_requests_seen = 1
- risk_aware_objective_requests_consumed = 1
- risk_aware_objective_requests_rejected = 0
- hypothesis_batches = 1
- hypotheses_generated = 15
- testable_hypotheses = 15
- ready_for_m3_g5_candidate_experiment_request = 15
- blocked_not_testable_hypotheses = 0
- final_invalid_hypotheses = 0
- all_requested_hypothesis_families_covered = true
- all_hypotheses_have_falsification_signal = true
- all_hypotheses_map_to_requested_experiment_style = true
- action6_extension_retest_hypotheses_generated = false
- substrate_actions_not_target_hypotheses = `ACTION6`, `ACTION6,ACTION3`, `ACTION6,ACTION4`
- execution_performed = false
- policy_rollout_performed = false
- environment_step_performed = false
- m3_write_performed = false
- a32_write_performed = false
- a33_write_performed = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_M2`
- wrong_confirmations = 0

Hypotheses produites :

- `objective_readiness_detection`
  - `H_readiness_relation_saturation` : la completion devient possible quand les nouveaux relation states cessent d'augmenter malgre un progres safe eleve.
  - `H_readiness_horizon_window` : il existe une fenetre d'horizon ou il faut basculer d'extension vers preparation/commit.
  - `H_readiness_state_geometry` : certains patterns spatiaux post-ACTION6 signalent une zone pre-completion.
- `post_conversion_commit_action_search`
  - `H_commit_after_safe_extension` : apres une conversion safe, une action de commit distincte peut declencher la completion.
  - `H_commit_requires_no_more_relation_progress` : le commit n'est utile qu'apres plateau du progres relationnel.
  - `H_commit_is_horizon_sensitive` : le commit devient necessaire avant la zone terminale mid/near.
- `goal_state_representation_beyond_safe_progress`
  - `H_goal_global_pattern_alignment` : le vrai but correspond a une configuration globale de la grille.
  - `H_goal_actor_target_contact` : la completion depend d'un contact acteur-cible specifique, pas du nombre total de relations.
  - `H_goal_region_or_boundary_condition` : le but est lie a une region, une bordure ou une condition spatiale globale.
- `proxy_progress_vs_completion_discriminator`
  - `H_proxy_completion_divergence_high_hold` : les etats high-hold peuvent scorer fort en proxy tout en restant mauvais ou risques pour completion.
  - `H_changed_pixels_false_friend` : beaucoup de changed pixels ne signifie pas readiness.
  - `H_relation_delta_false_friend` : un relation_delta eleve peut continuer sans declencher de completion.
- `risk_aware_selector_completion_gap`
  - `H_selector_missing_commit_branch` : le selector choisit hold/ACTION6/extensions mais n'a pas de branche commit.
  - `H_selector_target_wrong_metric` : le selector optimise terminal-adjusted progress au lieu d'une probabilite de completion.
  - `H_selector_needs_two_stage_policy` : il faut separer safe conversion puis objective commit.

Styles d'experience couverts :

- `post_selector_objective_readiness_probe`
- `post_conversion_commit_action_matrix`
- `terminal_safe_progress_vs_completion_discriminator`
- `horizon_conditioned_completion_trigger_search`
- `risk_aware_policy_ablation_with_completion_metrics`

Lecture :

M2.G2 transforme la frontier P2.G5 en hypotheses de completion. Le point cle
est le garde-fou `action6_extension_retest_hypotheses_generated=false` :
`ACTION6`, `ACTION6,ACTION3` et `ACTION6,ACTION4` restent des substrats de
contexte, pas les nouvelles hypotheses. Le probleme scientifique porte
maintenant sur le signal de readiness, l'action de commit, la representation du
but et le discriminateur entre progres safe et completion.

Suite naturelle :

- `M3.G5` - Objective-readiness / commit-action controlled experiment planner,
  qui devra compiler ces 15 hypotheses en tests controles sans inventer une
  nouvelle policy P3.

## M2.13a - State-conditioned local LLM generator

Objectif :

Brancher un LLM local comme source de propositions abductives strictement
encapsulee dans M2. Le LLM ne produit jamais de policy, de support, d'evidence
ni de verdict. Toute proposition brute repasse par le normalizer, le validator,
le merger et le testability compiler existants. Le backend reel `transformers`
est optionnel, offline et desactive par defaut ; un mock state-conditioned
deterministe est le fallback teste.

Entrees (resumes typés, lecture seule, jamais de grille brute) :

- `diagnostics/p2/risk_aware_objective_frontier_handoff_requests.json` (P2.G5)
- `diagnostics/m2/risk_aware_objective_completion_hypotheses.json` (M2.G2)
- `diagnostics/m1/general_mechanic_candidates.json` (M1.G0)
- `diagnostics/m3/risk_aware_objective_completion_experiment_results.json` (M3.G6)

Sorties :

- `diagnostics/m2/state_conditioned_llm_hypotheses.json`
- `diagnostics/m2/state_conditioned_llm_m3_candidate_requests.json`

Contrat :

- LLM `enable_local_llm=false` par defaut ; fallback mock obligatoire et teste.
- Sortie LLM JSON-only ; tout texte non JSON est rejete.
- Garde-fous boundary (avant normalizer) : rejet de `support>0`,
  `status=CONFIRMED/REFUTED`, `truth_status=CONFIRMED`, `ready_for_a32`,
  `ready_for_a33`, `revision_performed`, action cible indisponible, et retest
  substrat `ACTION6`/`ACTION6,ACTION3`/`ACTION6,ACTION4`.
- `candidate_action` doit etre dans `available_actions` ; `ACTION1/ACTION2/ACTION5`
  ne peuvent apparaitre que dans `unlock_target_actions` (separation stricte).
- Le packet nomme la section `mechanistic_context_candidates` (pas
  `world_model_candidates` : M2.14 fournira le vrai world model).
- Entites lues par `role_candidate` (timer_or_hud, controllable_actor, ...),
  jamais par identifiant code en dur.
- 6 familles candidates, dont `action_availability_or_unlocking_precondition`,
  bloquee si la metrique n'est pas encore mesurable par l'executor courant.
- Registry typé etendu (`metric_registry.py`) : `available_actions_before_after`
  (BLOCKED, `requires_env_frame_metadata`), `objective_completion_signal` et
  `terminal_reentry_rate` (READY, mesurees par M3.G5/G6). Regle a trois niveaux :
  unknown -> reject ; known-but-unmeasurable -> `BLOCKED_NOT_TESTABLE` ;
  known+supported -> `READY_FOR_M3`.
- Aucun write M3/A32/A33.

Run mock-fallback :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.local_llm_generator `
  --frontiers diagnostics\p2\risk_aware_objective_frontier_handoff_requests.json `
  --m2-hypotheses diagnostics\m2\risk_aware_objective_completion_hypotheses.json `
  --m1-candidates diagnostics\m1\general_mechanic_candidates.json `
  --m3g6-results diagnostics\m3\risk_aware_objective_completion_experiment_results.json `
  --out diagnostics\m2\state_conditioned_llm_hypotheses.json `
  --m3-out diagnostics\m2\state_conditioned_llm_m3_candidate_requests.json
```

Run vrai modele (B-pragmatique, fallback mock si absent) :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.m2.local_llm_generator `
  ... --enable-local-llm --model-path models\qwen2_5_3b_instruct `
  --device auto --max-new-tokens 768 --temperature 0.2
```

Resume du run mock-fallback :

- llm_enabled = false
- local_llm_backend = `mock`
- fallback_used = true
- local_llm_error = null
- hypotheses_generated = 6
- valid_hypotheses = 6
- hypothesis_families_covered = les 6 familles
- ready_for_m3_candidate_experiment_request = 5
- blocked_not_testable_hypotheses = 1 (unlock `available_actions_before_after`)
- direct_unavailable_action_hypotheses = 0
- substrate_retest_rejected = 0
- action6_extension_retest_hypotheses_generated = false
- support = 0
- revision_status = `CANDIDATE_ONLY`
- truth_status = `NOT_EVALUATED_BY_M2`
- a32_write_performed = false
- a33_write_performed = false
- wrong_confirmations = 0

L'hypothese unlock garde `candidate_action=ACTION3`, `unlock_target_actions=[ACTION2]`,
`metric=available_actions_before_after`, `testable=false`,
`blocking_reason=metric_known_but_unmeasurable_in_current_executor`.

Lecture :

M2.13a transforme l'etat/invariants candidats existants (M1.G0) et la
contradiction proxy/completion (M3.G6) en nouvelles hypotheses falsifiables
candidate-only, sans recopier les substrats ACTION6. Le LLM enrichit la file
M3 ; il n'est jamais cru. Hors scope : M2.14 (vrai world model) et M2.15 (fused).

Validation passee (pytest cibles verts, run CLI mock OK, deux artefacts generes,
guard anti-confirmation propre).
