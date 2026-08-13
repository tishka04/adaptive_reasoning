# SAGE.T12.4a.4d.1 — Runbook

Run these commands from the repository root in PowerShell. Commit the
implementation before freezing: the scientific freeze refuses a dirty tree.

## 1. Focused verification

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path

& $Py -m ruff check `
  .\theory\sage_t\causal\hazard_diversity_model.py `
  .\theory\sage_t\causal\hazard_diversity_protocol.py `
  .\theory\sage_t\causal\hazard_diversity_experiment.py `
  .\theory\sage_t\causal\hazard_diversity_cli.py `
  .\tests\test_sage_t_hazard_diversity_t12_4a_4d_1.py

& $Py -m pytest -q `
  .\tests\test_sage_t_hazard_diversity_t12_4a_4d_1.py `
  .\tests\test_sage_t_target_regrounding_t12_4a_4d.py `
  .\tests\test_sage_t_option_contract_t12_4a_4c.py `
  .\tests\test_sage_t_lineage_shield_t12_3e.py
```

## 2. Freeze

```powershell
$Parent = ".\training\sage_t\target_regrounding_t12_4a_4d_bp35"
$Root = ".\training\sage_t\hazard_diversity_t12_4a_4d_1_bp35"

& $Py -m theory.sage_t.causal.hazard_diversity_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\paired\target_regrounding_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.hazard_diversity_cli status `
  --manifest "$Root\manifest.json"
```

The status must authorize only `hazard_compile_authorized`. Active execution,
T12.4a.4e, validation, holdout and production must remain false.

## 3. Offline compile

```powershell
& $Py -m theory.sage_t.causal.hazard_diversity_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\compile"

& $Py -m theory.sage_t.causal.hazard_diversity_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json"
```

Continue only on `PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE` and
`hazard_diversity_active_run_authorized: true`. A compile miss is a scientific
stop, not permission to change the thresholds.

## 4. Prospective paired run

```powershell
& $Py -m theory.sage_t.causal.hazard_diversity_cli run `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

This executes 18 bounded arms: three fresh seeds, two exact lineages and three
policies. It may use at most 38,000 SDK calls and 3 GiB. Do not reuse a
non-empty output directory.

## 5. Adjudication

```powershell
& $Py -m theory.sage_t.causal.hazard_diversity_cli status `
  --manifest "$Root\manifest.json" `
  --compile-receipt "$Root\compile\compile_receipt.json" `
  --active-receipt "$Root\paired\hazard_diversity_receipt.json"
```

Only `PASS_T12_4A_4D_1_HAZARD_DIVERSITY_GATE` may open a separate T12.4a.4e
freeze. `guidance_claim_authorized` is a distinct, stricter causal-guidance
claim. On failure, retain the immutable artifacts and stop.

