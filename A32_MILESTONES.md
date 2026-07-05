# A32 milestones - journal de revision scientifique

Derniere mise a jour : 2026-06-19

Ce fichier suit la mise en place de A32, cote moteur scientifique. A32 consomme
la file candidate-only produite par M3.6 et produit une decision explicite dans
`diagnostics/a32`, sans modifier les artefacts M3.

## Principe fixe

M3 peut ecrire uniquement `CANDIDATE_ONLY`.

A32 est le premier endroit autorise a produire un verdict scientifique explicite
depuis la file M3.6. Les verdicts A32 sont confines a :

- `theory/a32/`
- `diagnostics/a32/`

A32 ne doit jamais ecrire de verdict dans :

- `theory/m3/`
- `diagnostics/m3/`

## Etat des milestones

| Milestone | Statut | Artefacts | Verification |
|---|---|---|---|
| A32.0 - Candidate intake preflight | Fait | `theory/a32_m3_revision_intake.py`, `tests/test_a32_m3_revision_intake.py`, `diagnostics/a32/m3_revision_intake.json` | File M3.6 consommable comme `HypothesisRecord` unresolved |
| A32.1 - Minimal A15 queue consumer | Fait | `theory/a32/revision_decisions.py`, `tests/test_a32_revision_decisions.py`, `diagnostics/a32/a15_revision_decisions.json` | 1 decision `REVISION_ACCEPTED_AS_CONFIRMED`, M3 non mute |
| A32.2 - Patch-similarity revision intake | Fait | `theory/a32/patch_similarity_revision_intake.py`, `tests/test_a32_patch_similarity_revision_intake.py`, `diagnostics/a32/patch_similarity_revision_intake.json` | 1 dossier M3.21 accepte pour revision scientifique; HypothesisRecord unresolved; no verdict; support=0 |
| A32.3 - Patch-similarity scientific revision decision | Fait | `theory/a32/patch_similarity_revision_decisions.py`, `tests/test_a32_patch_similarity_revision_decisions.py`, `diagnostics/a32/patch_similarity_revision_decisions.json` | Decision `SCOPE_LIMITED_CANDIDATE_ONLY`; tests supplementaires demandes; no A33; record unresolved |

## A32.0 - Candidate intake preflight

- Lit `diagnostics/m3/a15_revision_queue.json`.
- Verifie les garde-fous de file :
  - `status=UNRESOLVED`
  - `revision_status=CANDIDATE_ONLY`
  - `support=0`
  - `contradictions=0`
  - `controlled_test_required=true`
- Convertit l'entree en `HypothesisRecord` unresolved.
- Ne produit pas de verdict.

Resultat :

- `accepted_candidates=1`
- `candidate_records=1`
- `support=0`
- `hypotheses_confirmed=0`
- `hypotheses_refuted=0`
- `wrong_confirmations=0`

## A32.1 - Minimal A15 queue consumer

Objectif :

- Lire `diagnostics/m3/a15_revision_queue.json`.
- Consommer uniquement les hypotheses `mechanic_prediction::*`.
- Verifier que la file M3 a bien `support=0` avant revision.
- Verifier :
  - `support_events >= 3`
  - `independent_support_events >= 2`
  - `contradiction_events == 0`
  - au moins deux controles distincts avec support independant.
- Produire une decision explicite parmi :
  - `REVISION_ACCEPTED_AS_CONFIRMED`
  - `REVISION_REJECTED_AS_INSUFFICIENT`
  - `FOLLOWUP_REQUIRED`

Artefact :

- `diagnostics/a32/a15_revision_decisions.json`

Run du 2026-06-19 :

- entree : `diagnostics/m3/a15_revision_queue.json`
- sortie : `diagnostics/a32/a15_revision_decisions.json`
- `queue_items_consumed=1`
- `revision_accepted_as_confirmed=1`
- `revision_rejected_as_insufficient=0`
- `followup_required=0`
- `support_events=3`
- `independent_support_events=2`
- `reused_control_support_events=1`
- `contradiction_events=0`
- `wrong_confirmations=0`
- `m3_artifacts_mutated=false`

Decision produite :

- `REVISION_ACCEPTED_AS_CONFIRMED`
- raison : `a32_revision_criteria_satisfied`

Record d'entree :

- `status=unresolved`
- `support=0`
- `contradictions=0`
- `experiments_spent=3`

Record de decision :

- `status=confirmed`
- `support=2`
- `contradictions=0`
- `experiments_spent=3`

Lecture :

- La confirmation n'est pas produite par M3.
- La confirmation est produite par A32 apres consommation explicite de la file
  candidate M3.6.
- Le support retenu dans le `decision_record` est le support independant (`2`),
  pas le support brut (`3`), car une experience reutilise `ACTION3`.

## A32.2 - Patch-similarity revision intake

Objectif :

- Lire `diagnostics/m3/patch_similarity_a32_revision_queue.json`.
- Verifier les garde-fous du dossier generatif patch-similarity.
- Produire un `HypothesisRecord` unresolved.
- Decider uniquement `ACCEPTED_FOR_SCIENTIFIC_REVISION` ou
  `REJECTED_FROM_INTAKE`.
- Ne pas confirmer.
- Ne pas refuter.
- Ne pas ecrire A33.
- Ne pas muter M3.

Conditions d'acceptation :

- `status=UNRESOLVED`
- `revision_status=CANDIDATE_ONLY`
- `support=0`
- `ready_for_a32_revision=true`
- `ready_for_a32_revision_is_not_verdict=true`
- `source_success_metric_contradiction_events=0`
- `successful_args_total_count>=2`
- `failed_args_count>=1`
- `changed_pixels_role=effect_radar_not_success_metric`
- `diagnostic_contradictions_counted_as_refutation=false`

Rejets explicites :

- `support_must_remain_zero`
- `revision_status_not_candidate_only`
- `not_ready_for_a32_revision`
- `ready_for_a32_revision_not_marked_non_verdict`
- `success_metric_contradiction_events_present`
- `insufficient_successful_args`
- `missing_negative_case`
- `changed_pixels_role_not_diagnostic`
- `diagnostic_contradiction_interpreted_as_refutation`
- `a33_write_attempted`

Run du 2026-06-20 :

- entree :
  `diagnostics/m3/patch_similarity_a32_revision_queue.json`
- sortie :
  `diagnostics/a32/patch_similarity_revision_intake.json`
- `queue_items_consumed=1`
- `accepted_for_scientific_revision=1`
- `rejected_from_intake=0`
- `candidate_records=1`
- `successful_args_total_count=6`
- `failed_args_count=1`
- `source_success_metric_support_events=4`
- `source_success_metric_contradiction_events=0`
- `source_diagnostic_contradiction_events=2`
- `diagnostic_contradictions_counted_as_refutation=false`
- `changed_pixels_kept_diagnostic=true`
- `hypothesis_records_unresolved=1`
- `requested_next_step=A15_A31_PATCH_SIMILARITY_REVIEW_REQUIRED`
- `hypotheses_confirmed=0`
- `hypotheses_refuted=0`
- `revision_performed=false`
- `support=0`
- `truth_status=NOT_EVALUATED_BY_A32_INTAKE`
- `wrong_confirmations=0`
- `a33_write_performed=false`
- `m3_artifacts_mutated=false`

Candidat accepte :

- `intake_status=ACCEPTED_FOR_SCIENTIFIC_REVISION`
- key :
  `patch_similarity_rule::bp35-0a0ad940::ACTION4_ACTION6::local_patch_transformability`
- description :
  `ACTION4 after ACTION6/ACTION3 may open patch-similar ACTION6 affordances selected by local_patch_transformability.`
- context replay :
  `["ACTION6", "ACTION3", "ACTION4"]`
- context args :
  `[{"x": 18, "y": 0}, {}, {}]`
- succes totaux :
  - `{"x": 12, "y": 0}`
  - `{"x": 24, "y": 0}`
  - `{"x": 30, "y": 12}`
  - `{"x": 36, "y": 12}`
  - `{"x": 42, "y": 12}`
  - `{"x": 48, "y": 12}`
- echec connu :
  - `{"x": 30, "y": 0}`
- metriques de succes :
  - `local_patch_before_after`
  - `object_positions_before_after`
- metriques diagnostiques :
  - `changed_pixels`
  - `contact_graph_before_after`

Lecture :

- A32.2 accepte le dossier comme recevable pour revision scientifique.
- A32.2 ne decide pas que la mecanique est confirmee.
- La sortie conserve un `HypothesisRecord` unresolved avec `support=0`.
- La decision scientifique reste a faire dans une etape A32/A15-A31 ulterieure.

## A32.3 - Patch-similarity scientific revision decision

Objectif :

- Lire `diagnostics/a32/patch_similarity_revision_intake.json`.
- Produire une decision scientifique explicite parmi :
  - `CONFIRM_AFTER_SCOPE_LIMITED_REVISION`
  - `REFUTE_AFTER_REVISION`
  - `REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS`
  - `SCOPE_LIMITED_CANDIDATE_ONLY`
- Distinguer dossier local solide et mecanique generale A33-ready.
- Ne pas ecrire A33.
- Ne pas confirmer le dossier bp35 actuel comme mecanique generale.

Critere de confirmation scope-limited :

- Confirmation possible seulement si le dossier contient une validation
  multi-contexte ou multi-jeu explicite.
- Le dossier actuel est mono-jeu et mono-contexte :
  `bp35-0a0ad940`, `ACTION6 -> ACTION3 -> ACTION4`.
- Donc le dossier actuel reste `SCOPE_LIMITED_CANDIDATE_ONLY`.

Artefacts :

- `theory/a32/patch_similarity_revision_decisions.py`
- `A32PatchSimilarityRevisionDecision`
- `run_a32_patch_similarity_revision_decision_consumer`
- `diagnostics/a32/patch_similarity_revision_decisions.json`

Run du 2026-06-20 :

- entree :
  `diagnostics/a32/patch_similarity_revision_intake.json`
- sortie :
  `diagnostics/a32/patch_similarity_revision_decisions.json`
- `intake_candidates_consumed=1`
- `scope_limited_candidate_only=1`
- `recommended_more_tests=1`
- `request_more_tests_with_scope_limits=0`
- `confirm_after_scope_limited_revision=0`
- `refute_after_revision=0`
- `a33_ready_candidates=0`
- `a33_write_performed=false`
- `scientific_review_performed=true`
- `revision_performed=false`
- `confirmation_performed=false`
- `refutation_performed=false`
- `decision_records_unresolved=1`
- `decision_records_confirmed=0`
- `decision_records_refuted=0`
- `source_success_metric_support_events=4`
- `source_success_metric_contradiction_events=0`
- `diagnostic_contradictions_counted_as_refutation=false`
- `wrong_confirmations=0`

Decision produite :

- `SCOPE_LIMITED_CANDIDATE_ONLY`
- `recommended_next_step=REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS`
- raisons :
  - `mono_game_scope`
  - `mono_context_scope`
  - `scope_not_a33_ready`

Record d'entree :

- `status=unresolved`
- `support=0`
- `contradictions=0`
- `experiments_spent=8`

Record de decision :

- `status=unresolved`
- `support=0`
- `contradictions=0`
- `experiments_spent=8`

Tests demandes :

- `outside_known_y12_region_probe`
  - tester si la similarite de patch tient hors de la ligne / region de succes
    connue.
- `alternate_repositioning_context_probe`
  - tester si `ACTION4` cree des affordances `ACTION6` patch-similar dans un
    autre contexte de replay.

Lecture :

- A32.3 ne rejette pas le dossier : il le classe comme hypothese locale solide.
- A32.3 ne le confirme pas non plus : le scope est trop etroit pour A33.
- La suite propre est de convertir les tests demandes en nouvelles requests M3,
  puis de revenir vers A32 avec un scope elargi ou une contradiction de succes.

## Commandes de verification

Tests A32 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_a32_m3_revision_intake.py tests\test_a32_revision_decisions.py tests\test_a32_patch_similarity_revision_intake.py tests\test_a32_patch_similarity_revision_decisions.py -q
```

Generation decision A32 :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a32.revision_decisions --queue diagnostics\m3\a15_revision_queue.json --out diagnostics\a32\a15_revision_decisions.json
```

A32.2 intake patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a32.patch_similarity_revision_intake --queue diagnostics\m3\patch_similarity_a32_revision_queue.json --out diagnostics\a32\patch_similarity_revision_intake.json
```

A32.3 decision patch-similarity :

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.a32.patch_similarity_revision_decisions --intake diagnostics\a32\patch_similarity_revision_intake.json --out diagnostics\a32\patch_similarity_revision_decisions.json
```

Guard M3 :

- Verifier que les marqueurs de verdict A32 ne sont pas presents dans
  `theory/m3`, `diagnostics/m3`, `M3_MILESTONES.md`.
