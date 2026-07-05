# HUD Milestones

HUD valide les variables perceptives transverses avant qu'elles alimentent une policy. Les resultats restent candidate-only et ne produisent pas de verdict scientifique.

## HUD.2 - Real visual history validation

Status: implemented.

Files:
- `theory/p3/terminal_horizon_hud_validation.py`
- `diagnostics/p3/bp35_terminal_horizon_hud_validation.json`

Contract:
- Capture a real bp35 visual history under `patch_similarity_soft_stale_action6_prefix`.
- Call `TerminalHorizonObserver` at each step with the grid history.
- Log source, confidence, estimated moves remaining, fraction remaining, HUD bbox/orientation when present, monotonicity score, predicted next remaining, and actual terminal proxy.
- Keep `empirical_fallback` active when no stable HUD bar is detected.
- Do not read A33, LLM, or world model.
- Do not modify M2, M3, A32, or A33.
- Keep `support=0`, `revision_status=CANDIDATE_ONLY`, `truth_status=NOT_EVALUATED_BY_HUD_VALIDATION`.

Interpretation rules:
- `source=hud_bar` is valid only if a stable monotone edge segment is selected across the real history.
- `source=empirical_fallback` is an honest fallback, not a failure of the observer.
- A HUD detection result is never counted as confirmation.

Run summary:
- Real bp35 history captured: 65 grid states, 64 actions, terminal `GAME_OVER`.
- `hud_bar_source_active_steps=61`.
- `empirical_fallback_steps=4`.
- Dominant HUD bbox: `[63, 0, 63, 63]`.
- Orientation: `horizontal_bottom`.
- Dominant bbox stable steps: 61.
- `hud_estimated_remaining_nonincreasing=true`.
- `stable_hud_bar_sequence_detected=true`.
- `ready_for_hud_p3_2b=true`.
- `support=0`.

Reading:
- bp35 exposes a reliable bottom HUD bar on the real visual trajectory.
- The bar is elapsed/action-used style: filled cells increase by one per action.
- The observer converts it to remaining moves with `remaining = length - filled_length`.
- At step 63, `estimated_moves_remaining=1`; at terminal step 64, `estimated_moves_remaining=0`.
- This validates `hud_bar` as a usable source for a P3.2b rerun, without treating it as scientific confirmation.

P3.2b handoff:
- `diagnostics/p3/bp35_terminal_horizon_hud_policy_probe.json` consumes the HUD.2 validation path.
- Candidate policy runs using the observed HUD source: 72/72.
- Horizon triggers using `source=hud_bar`: 72/72.
- Empirical fallback remains available for warmup, but is not the candidate decision source.
- `objective_completion_signal_runs=0`, so HUD.2 provides perception for terminal avoidance, not objective completion.
