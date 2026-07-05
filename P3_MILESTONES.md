# P3 Milestones

P3 teste l'utilite agentique de regles candidate-only issues de M3/A32 sans confirmer de mecanique, sans ecrire A33, et sans compter un resultat policy comme preuve scientifique.

## P3.1 - Terminal-aware stop policy probe

Status: implemented.

Inputs:
- `diagnostics/m3/objective_refined_window_results.json`
- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`

Output:
- `diagnostics/p3/bp35_terminal_aware_stop_policy_probe.json`

Contract:
- Compare `patch_similarity_soft_stale_action6_prefix` against `terminal_aware_stop_at_threshold`.
- Derive the stop threshold from M3.O4 candidate-only terminal avoidance.
- Treat stop/hold as terminal avoidance, not objective completion.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not read A33, LLM, or world model.
- Do not modify M2, M3, A32, or A33.

Interpretation rules:
- `terminal_avoidance_signal` means the candidate avoids `GAME_OVER` where the baseline does not.
- `objective_completion_signal` requires improved completed levels.
- `terminal_avoidance_only` is not counted as solving bp35.
- `candidate_policy_counted_as_confirmation=false`.
- `policy_result_counted_as_scientific_verdict=false`.

Run summary:
- Budgets tested: 64, 96, 128.
- Tie-break seeds tested: 0, 1, 2.
- Baseline `GAME_OVER` runs: 9/9.
- Terminal-aware candidate `GAME_OVER` runs: 0/9.
- `terminal_avoidance_signal_runs=9`.
- `objective_completion_signal_runs=0`.
- `terminal_avoidance_only_runs=9`.

Reading:
- Stop/hold at the M3.O4 threshold avoids the observed terminal outcome.
- This is not bp35 completion and is not a scientific verdict.
- The next useful question is an objective-seeking post-stop policy, not more local ACTION6 exploitation.

## P3.2 - Terminal horizon state variable and objective mode probe

Status: implemented.

Files:
- `theory/p3/terminal_horizon_estimator.py`
- `theory/p3/terminal_horizon_policy_probe.py`
- `diagnostics/p3/bp35_terminal_horizon_policy_probe.json`

Contract:
- Replace the hard `ACTION6` prefix threshold with `estimated_moves_remaining`.
- Count all consumed environment actions through `moves_used`, not only `ACTION6`.
- Use `TerminalHorizonObserver` with fusion priority:
  `environment_metadata -> hud_bar -> empirical_fallback -> unknown`.
- Start bp35 runtime with `source=empirical_fallback` and `terminal_budget_estimate=64`.
- Keep API slots and tests for `environment_metadata` and `hud_bar`.
- Detect simple HUD bars as monotone edge segments across history.
- Log `terminal_fraction_remaining`, `terminal_horizon_evidence`, and `horizon_trigger_log`.
- Test `k in {1, 2, 4, 6, 8, 12}`.
- Compare `terminal_horizon_stop_guard` and `terminal_horizon_objective_mode`.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not read A33, LLM, or world model.
- Do not modify M2, M3, A32, or A33.

Run summary:
- Budgets tested: 64, 96.
- Tie-break seeds tested: 0, 1, 2.
- Baseline runs: 6.
- Candidate policy runs: 72.
- `terminal_horizon_estimator_integrated=true`.
- `TerminalHorizonObserver` fusion priority is recorded.
- `action6_prefix_count_used_as_decision_variable=false`.
- `terminal_fraction_remaining_available=true`.
- `evidence_tracking_enabled=true`.
- `hud_bar_detector_available=true`.
- `terminal_avoidance_signal_runs=72`.
- `objective_mode_entered_runs=30`.
- `objective_completion_signal_runs=0`.
- `terminal_avoidance_only_runs=72`.

Reading:
- `estimated_moves_remaining` is now the policy state variable.
- In the current bp35 run, the active source is still `empirical_fallback`.
- The observer can already accept perfect metadata and detect synthetic monotone HUD bars.
- Every tested horizon guard avoids `GAME_OVER` relative to baseline.
- The naive objective mode enters before the final stop for `k>1`, but does not complete bp35.
- Objective mode preserves more progress than stopping early at large `k`, but remains only terminal-avoidance/objective-search candidate behavior.
- The next useful step is a stronger objective-conversion policy, not another hard stop threshold.

HUD.2 update:
- Real bp35 visual history validates `hud_bar` as a stable source after the initial history warmup.
- Dominant HUD bbox: `[63, 0, 63, 63]`, orientation `horizontal_bottom`.
- The bar is elapsed-style, so the observer uses `remaining = length - filled_length`.
- P3.2b can rerun the horizon policy with observed `source=hud_bar` rather than `empirical_fallback`.

## P3.2b - Terminal horizon policy with observed HUD source

Status: implemented.

Files:
- `theory/p3/terminal_horizon_hud_policy_probe.py`
- `diagnostics/p3/bp35_terminal_horizon_hud_policy_probe.json`

Inputs:
- `diagnostics/m3/objective_refined_window_results.json`
- `diagnostics/m3/a32_requested_patch_similarity_scope_consolidation.json`
- `diagnostics/p3/bp35_terminal_horizon_hud_validation.json`

Contract:
- Rerun the P3.2 terminal horizon policy using observed `source=hud_bar` after HUD.2 validation.
- Keep warmup fallback available, but require decisions/triggers to record their actual source.
- Keep `action6_prefix_count_used_as_decision_variable=false`.
- Keep `terminal_avoidance_signal` separate from `objective_completion_signal`.
- Keep `terminal_avoidance_only` out of success/completion accounting.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not read A33, LLM, or world model.
- Do not modify M2, M3, A32, or A33.

Run summary:
- Budgets tested: 64, 96.
- Tie-break seeds tested: 0, 1, 2.
- Candidate policy runs: 72.
- `candidate_hud_bar_source_runs=72`.
- `hud_bar_trigger_source_runs=72`.
- `candidate_empirical_fallback_source_runs=0`.
- `terminal_avoidance_signal_runs=72`.
- `objective_mode_entered_runs=30`.
- `objective_completion_signal_runs=0`.
- `terminal_avoidance_only_runs=72`.
- `action6_prefix_count_used_as_decision_variable=false`.
- `support=0`.

Reading:
- P3.2's behavior is preserved when the terminal horizon is perceived from the real bp35 HUD bar.
- The decision variable is now `estimated_moves_remaining` from `hud_bar`, not the bp35-specific ACTION6 prefix count.
- The observed HUD evidence uses bbox `[63, 0, 63, 63]`, `horizontal_bottom`, elapsed-style semantics, and `remaining = length - filled_length`.
- This validates the perceptive path into P3, but still only produces terminal avoidance.
- `objective_completion_signal_runs=0`, so the next policy problem remains objective conversion rather than another stop guard.

## P3.G0 - Abstract-mechanic policy probe

Status: implemented.

Files:
- `theory/p3/abstract_mechanic_policy_probe.py`
- `tests/test_p3_abstract_mechanic_policy_probe.py`
- `diagnostics/p3/abstract_mechanic_policy_adapter.json`
- `diagnostics/p3/abstract_mechanic_policy_probe.json`
- `diagnostics/p3/abstract_mechanic_policy_utility_consolidation.json`

Inputs:
- `diagnostics/m3/generic_candidate_symbolic_mechanism_model.json`

Contract:
- Consume the M3.G0.7 symbolic mechanism model as candidate-only policy material.
- Build a bounded adapter around actor candidate `E182`, action candidates `ACTION3`/`ACTION4`, relation targets `E136`/`E137`/`E138`/`E139`, and dynamic invariant caveats.
- Compare against random available action, greedy changed-pixels, and terminal-horizon guard baselines.
- Measure policy utility, actor/relation usage, terminal behavior, and objective completion separately.
- Keep `candidate_model_counted_as_confirmed_mechanic=false`.
- Keep `policy_result_counted_as_scientific_verdict=false`.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not write A32 or A33.

Run summary:
- Adapter: `actor_candidates=1`, `action_candidates=2`, `relation_targets=4`, `ignored_or_caveated_entities=1`.
- Budgets tested: 8, 16.
- Tie-break seeds tested: 0, 1, 2.
- Rollout runs: 24.
- Candidate mean progress proxy: 120.0.
- Best baseline mean progress proxy: 117.5.
- Candidate mean actor relation delta count: 12.0.
- Best baseline mean actor relation delta count: 11.5.
- Candidate beats best baseline progress: true.
- Candidate beats best baseline relation metric: true.
- Policy utility status: `POLICY_USEFUL_CANDIDATE_ONLY`.
- Candidate levels completed: 0.0 mean.
- Candidate terminal rate: 0.5.
- `support=0`.

Reading:
- The M3.G0.7 symbolic model is useful as a candidate policy prior in this short bp35 probe.
- The gain is modest and appears on progress/relation metrics, not objective completion.
- Terminal behavior is not improved relative to greedy changed-pixels, so this is not an objective-solving policy.
- The result is an agentic utility signal only, not a scientific confirmation of E182, ACTION3/ACTION4, E193, or any relation target.
- The next useful step is a P3.G1 objective-aware abstract policy that combines relation progress with terminal-horizon risk instead of optimizing relation progress alone.

## P3.G1 - Objective-aware abstract mechanic policy

Status: implemented.

Files:
- `theory/p3/objective_aware_abstract_policy_probe.py`
- `tests/test_p3_objective_aware_abstract_policy_probe.py`
- `diagnostics/p3/objective_aware_abstract_policy_adapter.json`
- `diagnostics/p3/objective_aware_abstract_policy_probe.json`
- `diagnostics/p3/objective_aware_abstract_policy_utility_consolidation.json`

Inputs:
- `diagnostics/p3/abstract_mechanic_policy_adapter.json`
- `diagnostics/p3/bp35_terminal_horizon_hud_policy_probe.json`
- `diagnostics/m3/generic_candidate_symbolic_mechanism_model.json`

Contract:
- Combine the M3.G0 symbolic policy prior with observed HUD terminal horizon.
- Test `lambda_terminal_risk in {0, 1, 3, 10, 30}`.
- Compare random available action, greedy changed-pixels, terminal-horizon guard, P3.G0 abstract policy, and P3.G1 objective-aware variants.
- Add `terminal_adjusted_progress = progress_proxy if not terminal else 0`.
- Keep `policy_result_counted_as_scientific_verdict=false`.
- Keep `candidate_model_counted_as_confirmed_mechanic=false`.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not write A32 or A33.

Run summary:
- Budgets tested: 8, 16, 32, 64.
- Tie-break seeds tested: 0, 1, 2, 3, 4.
- Rollout runs: 180.
- P3.G0 mean progress proxy: 140.0.
- P3.G0 terminal rate: 0.75.
- P3.G0 terminal-adjusted progress: 20.0.
- Best P3.G1 condition: `objective_aware_abstract_policy_lambda_0`.
- Best P3.G1 mean progress proxy: 132.5.
- Best P3.G1 terminal rate: 0.0.
- Best P3.G1 terminal-adjusted progress: 132.5.
- Objective completion signal: false.
- Policy utility status: `POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY`.
- `support=0`.

Reading:
- Adding terminal-aware lookahead prevents the terminal outcomes produced by P3.G0 in this probe.
- This safety comes with lower raw relation/progress exploitation: 132.5 vs 140.0.
- No tested lambda produces objective completion.
- The terminal lookahead dominates the lambda sweep in the current setup: all objective-aware variants converge to the same aggregate behavior.
- P3.G1 is therefore a safer candidate policy, not a solver.
- The next useful question is not more terminal penalty, but an objective-conversion action/model once terminal-risk avoidance has stopped the relation-only policy.

## P3.G2 - Contextual post-stop conversion policy probe

Status: implemented.

Files:
- `theory/p3/contextual_post_stop_conversion_policy_probe.py`
- `tests/test_p3_contextual_post_stop_conversion_policy_probe.py`
- `diagnostics/p3/contextual_post_stop_conversion_policy_adapter.json`
- `diagnostics/p3/contextual_post_stop_conversion_policy_probe.json`

Inputs:
- `diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json`
- `diagnostics/p3/abstract_mechanic_policy_probe.json`
- `diagnostics/p3/objective_aware_abstract_policy_utility_consolidation.json`

Contract:
- Consume M3.G4 as policy material, not as mechanic confirmation.
- Choose among `hold_or_stop_state`, `ACTION6`, `ACTION6,ACTION3`, and `ACTION6,ACTION4`.
- Default to `ACTION6` when it is nonterminal and beats hold.
- Allow two-action extensions only when the `sampling_family + terminal_horizon_band` scope and the `hold_baseline_band` did not show terminal re-entry in M3.G4.
- Require an extension to beat `ACTION6` by at least 5 terminal-adjusted progress points.
- Report terminal rate vs P3.G0/P3.G1 references, mean terminal-adjusted progress, improvement over `ACTION6`, unsafe extensions avoided, objective completion, and hold/abstention rate.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not write A32 or A33.

Run summary:
- Source M3.G4 status: `MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY`.
- Safe-stops evaluated: 13.
- Selected extensions: 9.
- Selected `ACTION6` fallback: 4.
- Selected hold/abstention: 0.
- Contextual policy mean terminal-adjusted progress: 80.0.
- Hold mean terminal-adjusted progress: 68.076923.
- `ACTION6` mean terminal-adjusted progress: 73.076923.
- Always `ACTION6,ACTION3` mean terminal-adjusted progress: 71.153846, terminal rate 0.076923.
- Always `ACTION6,ACTION4` mean terminal-adjusted progress: 71.153846, terminal rate 0.076923.
- Mean delta vs hold: 11.923077.
- Mean delta vs `ACTION6`: 6.923077.
- Terminal rate: 0.0.
- Terminal-rate delta vs P3.G0 abstract policy: -0.5.
- Terminal-rate delta vs P3.G1 objective-aware policy: 0.0.
- Unsafe extension options avoided: 2.
- Unsafe extension safe-stops avoided: 1.
- Objective completion signal: false.
- Policy utility status: `POST_STOP_CONTEXTUAL_POLICY_CANDIDATE_ONLY`.
- `support=0`.

Reading:
- P3.G2 turns M3.G4's scope map into a candidate-only decision rule.
- `ACTION6` is the safe fallback nucleus after stop.
- `ACTION6,ACTION3` and `ACTION6,ACTION4` are not universal best candidates; they are conditional accelerators.
- The current selector avoids the observed terminal extension case while preserving most extension gains outside the risky family/horizon/hold scope.
- This is policy utility only, not proof that ACTION6 is a mechanic or that the extension rule generalizes beyond the M3.G4 safe-stop substrate.
- The next useful frontier is to decide whether this selector should be stress-tested on fresh safe-stop acquisitions or handed to P2 as a new safe-weak-vs-risky-medium policy frontier.

Command:
```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p3.contextual_post_stop_conversion_policy_probe --stage all --source-m3g4 diagnostics\m3\objective_conversion_diverse_safe_stop_validation.json --adapter-out diagnostics\p3\contextual_post_stop_conversion_policy_adapter.json --probe-out diagnostics\p3\contextual_post_stop_conversion_policy_probe.json --p3g0-policy-probe diagnostics\p3\abstract_mechanic_policy_probe.json --p3g1-consolidation diagnostics\p3\objective_aware_abstract_policy_utility_consolidation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_p3_abstract_mechanic_policy_probe.py tests\test_p3_objective_aware_abstract_policy_probe.py tests\test_p3_contextual_post_stop_conversion_policy_probe.py -q
```

## P3.G3 - Out-of-sample contextual post-stop policy validation

Status: implemented.

Files:
- `theory/p3/out_of_sample_contextual_post_stop_policy_validation.py`
- `tests/test_p3_out_of_sample_contextual_post_stop_policy_validation.py`
- `diagnostics/p3/out_of_sample_contextual_post_stop_policy_validation.json`

Inputs:
- `diagnostics/p3/contextual_post_stop_conversion_policy_adapter.json`
- `diagnostics/p3/contextual_post_stop_conversion_policy_probe.json`
- `diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json`

Contract:
- Consume the P3.G2 adapter exactly as frozen policy material.
- Do not relearn gates from out-of-sample results.
- Generate candidate safe-stops whose state/prefix was not used by M3.G4/P3.G2.
- Keep only replay-exact, non-terminal, terminal-safe safe-stops with measurable hold baseline.
- Apply the contextual selector using only frozen adapter gates and safe-stop features.
- Execute the selected option and the static baselines: hold, `ACTION6`, always `ACTION6,ACTION3`, always `ACTION6,ACTION4`.
- Measure terminal rate, terminal-adjusted progress, delta vs `ACTION6`, unsafe extensions avoided, and objective completion.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not write A32 or A33.

Run summary:
- Candidate plans: 19.
- Candidate plans executed: 19.
- Accepted out-of-sample safe-stops: 14.
- In-sample safe-stops rejected: 1.
- Duplicate out-of-sample safe-stops rejected: 3.
- Unsafe/terminal safe-stops rejected: 1.
- Source cells rerun: true.
- Adapter relearned: false.
- Selection uses out-of-sample candidate outcomes: false.
- Cells executed: 56.
- Selected extensions: 12.
- Selected `ACTION6` fallback: 2.
- Selected hold/abstention: 0.
- Contextual policy terminal rate: 0.0.
- Contextual policy mean terminal-adjusted progress: 78.928571.
- Hold mean terminal-adjusted progress: 65.357143.
- `ACTION6` mean terminal-adjusted progress: 70.357143.
- Always `ACTION6,ACTION3` mean terminal-adjusted progress: 80.357143, terminal rate 0.0.
- Always `ACTION6,ACTION4` mean terminal-adjusted progress: 80.357143, terminal rate 0.0.
- Mean delta vs hold: 13.571429.
- Mean delta vs `ACTION6`: 8.571428.
- Unsafe extension options avoided: 0.
- Objective completion signal: false.
- Policy utility status: `OUT_OF_SAMPLE_POLICY_UTILITY_CANDIDATE_ONLY`.
- `support=0`.

Reading:
- P3.G3 confirms that the frozen contextual selector has out-of-sample policy utility on fresh safe-stops: it beats hold and `ACTION6` while keeping terminal rate at 0.
- The result does not justify the stronger "generalizes as dominant selector" reading, because in this particular out-of-sample slice the always-on two-action extensions also stay terminal-safe and score slightly higher.
- The conservative gates cost progress on two high-hold safe-stops, but do not create terminal risk.
- No objective completion appears.
- This is still a candidate-only agentic result, not a mechanic confirmation and not a scientific verdict.
- The next frontier is now sharper: decide whether to relax the high-hold/horizon gate, or to keep the conservative selector and search for out-of-sample contexts where static extensions again become risky.

Command:
```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p3.out_of_sample_contextual_post_stop_policy_validation --adapter diagnostics\p3\contextual_post_stop_conversion_policy_adapter.json --source-m3g4 diagnostics\m3\objective_conversion_diverse_safe_stop_validation.json --source-p3g2-probe diagnostics\p3\contextual_post_stop_conversion_policy_probe.json --out diagnostics\p3\out_of_sample_contextual_post_stop_policy_validation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_p3_abstract_mechanic_policy_probe.py tests\test_p3_objective_aware_abstract_policy_probe.py tests\test_p3_contextual_post_stop_conversion_policy_probe.py tests\test_p3_out_of_sample_contextual_post_stop_policy_validation.py -q
```

## P3.G4 - Risk-targeted OOS contextual post-stop policy validation

Status: implemented.

Files:
- `theory/p3/risk_targeted_contextual_post_stop_policy_validation.py`
- `tests/test_p3_risk_targeted_contextual_post_stop_policy_validation.py`
- `diagnostics/p3/risk_targeted_contextual_post_stop_policy_validation.json`

Inputs:
- `diagnostics/p3/contextual_post_stop_conversion_policy_adapter.json`
- `diagnostics/p3/contextual_post_stop_conversion_policy_probe.json`
- `diagnostics/p3/out_of_sample_contextual_post_stop_policy_validation.json`
- `diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json`

Contract:
- Consume the P3.G2 adapter exactly as frozen policy material.
- Exclude safe-stop state/prefix identities already used by M3.G4 and P3.G3.
- Target fresh OOS safe-stops near the known risk region: high hold baseline and/or mid/near terminal horizon.
- Keep only replay-exact, non-terminal, terminal-safe safe-stops with measurable hold baseline.
- Apply the contextual selector using only frozen adapter gates and safe-stop features.
- Execute the selected option and the static baselines: hold, `ACTION6`, always `ACTION6,ACTION3`, always `ACTION6,ACTION4`.
- Measure whether static extensions re-enter terminal and whether the selector avoids those extensions.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_P3_AGENT_PROBE`.
- Do not write A32 or A33.

Run summary:
- Candidate plans: 48.
- Candidate plans executed: 48.
- Accepted risk-targeted safe-stops: 7.
- Duplicate risk-targeted safe-stops rejected: 29.
- Unsafe/terminal safe-stops rejected: 12.
- Source cells rerun: true.
- Adapter relearned: false.
- Selection uses risk-targeted candidate outcomes: false.
- Cells executed: 28.
- Selected extensions: 2.
- Selected `ACTION6` fallback: 5.
- Selected hold/abstention: 0.
- Contextual policy terminal rate: 0.0.
- Contextual policy mean terminal-adjusted progress: 142.857143.
- Hold mean terminal-adjusted progress: 135.0.
- `ACTION6` mean terminal-adjusted progress: 140.0.
- Always `ACTION6,ACTION3` mean terminal-adjusted progress: 104.285714, terminal rate 0.285714.
- Always `ACTION6,ACTION4` mean terminal-adjusted progress: 104.285714, terminal rate 0.285714.
- Mean delta vs hold: 7.857143.
- Mean delta vs `ACTION6`: 2.857143.
- Static extension terminal options: 4.
- Static extension terminal safe-stops: 2.
- Unsafe extension options avoided by selector: 4.
- Objective completion signal: false.
- Policy utility status: `RISK_AWARE_OOS_POLICY_UTILITY_CANDIDATE_ONLY`.
- `support=0`.

Reading:
- P3.G4 answers the question left open by P3.G3: the static-extension risk was not only an in-sample M3.G4 artifact.
- The risk reappears out-of-sample in the targeted region `hold_high_ge_120 + horizon_mid_45_54`.
- Both always-on two-action extensions re-enter terminal on 2/7 targeted safe-stops.
- The frozen contextual selector avoids all 4 terminal extension options by falling back to `ACTION6`, keeps terminal rate at 0, and still beats both hold and `ACTION6`.
- This supports the conservative gate as risk-aware policy utility, not as a mechanic confirmation.
- No objective completion appears, and the result remains candidate-only with no A32/A33 write.
- The next useful question is no longer whether the gate has any purpose; it is how to relax it without losing the two OOS terminal-avoidance wins.

Command:
```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.p3.risk_targeted_contextual_post_stop_policy_validation --adapter diagnostics\p3\contextual_post_stop_conversion_policy_adapter.json --source-m3g4 diagnostics\m3\objective_conversion_diverse_safe_stop_validation.json --source-p3g2-probe diagnostics\p3\contextual_post_stop_conversion_policy_probe.json --source-p3g3 diagnostics\p3\out_of_sample_contextual_post_stop_policy_validation.json --out diagnostics\p3\risk_targeted_contextual_post_stop_policy_validation.json
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest tests\test_p3_abstract_mechanic_policy_probe.py tests\test_p3_objective_aware_abstract_policy_probe.py tests\test_p3_contextual_post_stop_conversion_policy_probe.py tests\test_p3_out_of_sample_contextual_post_stop_policy_validation.py tests\test_p3_risk_targeted_contextual_post_stop_policy_validation.py -q
```
