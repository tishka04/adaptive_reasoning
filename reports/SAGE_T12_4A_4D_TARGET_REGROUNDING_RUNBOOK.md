# SAGE.T12.4a.4d — Runbook

Run from the repository root in PowerShell after committing the implementation.
The scientific freeze refuses a dirty worktree.

## 1. Local verification

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path

& $Py -m ruff check `
  .\theory\sage_t\causal\target_regrounding_protocol.py `
  .\theory\sage_t\causal\target_regrounding_experiment.py `
  .\theory\sage_t\causal\target_regrounding_cli.py `
  .\tests\test_sage_t_target_regrounding_t12_4a_4d.py

& $Py -m pytest -q `
  .\tests\test_sage_t_target_regrounding_t12_4a_4d.py `
  .\tests\test_sage_t_option_contract_t12_4a_4c.py `
  .\tests\test_sage_t_option_applicability_t12_4a_4b.py `
  .\tests\test_sage_t_option_transfer_t12_4a_4.py `
  .\tests\test_sage_t_lineage_shield_t12_3e.py
```

## 2. Freeze

```powershell
$Parent = ".\training\sage_t\option_contract_t12_4a_4c_bp35"
$Root = ".\training\sage_t\target_regrounding_t12_4a_4d_bp35"

& $Py -m theory.sage_t.causal.target_regrounding_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\contract\option_contract_receipt.json" `
  --witness-registry ".\training\sage_t\witness_reconfirmation_t12_4a_2_bp35\witnesses.sealed.json" `
  --shield-manifest ".\training\sage_t\lineage_shield_t12_3e_bp35\manifest.json" `
  --shield-receipt ".\training\sage_t\lineage_shield_t12_3e_bp35\paired\lineage_shield_receipt.json" `
  --manifest "$Root\manifest.json"
```

Inspect the freeze before spending SDK calls:

```powershell
& $Py -m theory.sage_t.causal.target_regrounding_cli status `
  --manifest "$Root\manifest.json"
```

Expected pre-run state: the target-regrounding experiment is authorized, while
T12.4a.4e, neural control, validation, holdout and production remain closed.

## 3. Paired physical run

```powershell
& $Py -m theory.sage_t.causal.target_regrounding_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\paired" `
  --environments-dir .\environment_files
```

This command performs twelve bounded search arms: three fresh seeds, two exact
route lineages and two paired policies. It then confirms the first
preregistered discovery twice from each lineage. It never writes raw frames and
hard-fails above 3 GiB.

Do not rerun into the same output directory. A new scientific run requires a
new frozen root.

## 4. Adjudication

```powershell
& $Py -m theory.sage_t.causal.target_regrounding_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\paired\target_regrounding_receipt.json"
```

Interpret the two results separately:

- `PASS_T12_4A_4D_TARGET_WITNESS_GATE` authorizes only the T12.4a.4e freeze;
- `guidance_claim_authorized: true` is additionally required to claim that
  contract reranking outperformed generic local Go-Explore.

On `FAIL_T12_4A_4D_TARGET_WITNESS_GATE`, retain the immutable report and stop.
Do not change thresholds or open validation/holdout data under this manifest.

Principal artifacts are:

- `paired/target_regrounding_receipt.json`;
- `paired/target_regrounding_report.json`;
- `paired/progress_witnesses.sealed.json`;
- `paired/confirmation_trials.json`;
- `paired/intervention_bundles.json`;
- per-seed/per-lineage paired symbolic archives.
