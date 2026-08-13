# SAGE.T12.5 runbook

Run from the repository root in PowerShell after committing the implementation.
The freeze requires a clean worktree. Every T12.5 phase below is offline: it
makes zero ARC SDK calls and the artifact directory is capped at 3 GiB.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Contract = ".\training\sage_t\option_contract_t12_4a_4c_bp35"
$Ablation = ".\training\sage_t\option_minimization_t12_4a_3r1_bp35"
$Root = ".\training\sage_t\causal_progress_t12_5_bp35"

& $Py -m theory.sage_t.causal.progress_experiment_cli freeze `
  --parent-manifest "$Contract\manifest.json" `
  --parent-receipt "$Contract\contract\option_contract_receipt.json" `
  --ablation-receipt "$Ablation\ablation\option_ablation_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.progress_experiment_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\compiled"

& $Py -m theory.sage_t.causal.progress_experiment_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\compiled\causal_progress_receipt.json"
```

The successful status is `PASS_T12_5_CAUSAL_PROGRESS_GATE`. Inspect these
fields before doing anything else:

- `ordered_replication_accuracy` must equal 1;
- `posterior_replication_accuracy` must equal 1;
- `posterior_mass_by_kind.ordered_effects` must be at least 0.95;
- every entry under `metrics.checks` must be true;
- `maximum_parent_mass_error` must be at most `1e-12`;
- `storage.within_budget` must be true;
- every control, validation, holdout and production authority must remain
  false.

A failure is a scientific stop, not permission to lower the threshold or add
the replication lineage to induction retroactively. A pass opens only a new,
separately checksummed shadow-ranking freeze. It does not execute ARC, change
the live policy or demonstrate transfer to later levels.

