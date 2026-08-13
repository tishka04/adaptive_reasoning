# SAGE.T12.4a.4c runbook

Run from the repository root in PowerShell after committing the implementation.
The freeze requires a clean worktree. The compile phase is entirely offline and
makes zero ARC SDK calls.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\option_applicability_t12_4a_4b_bp35"
$Root = ".\training\sage_t\option_contract_t12_4a_4c_bp35"

& $Py -m theory.sage_t.causal.option_contract_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\audit\applicability_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.option_contract_cli compile `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\contract"

& $Py -m theory.sage_t.causal.option_contract_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\contract\option_contract_receipt.json"
```

The successful status is `PASS_T12_4A_4C_OPTION_CONTRACT_GATE`. It means the
option is correctly guarded and compiled in shadow, not that it transfers or
controls the environment. A pass authorizes only the freeze of a target-local
re-grounding/search experiment for the failed level-1 context.

The output directory is immutable and bounded to 3 GiB. Do not reuse an
existing directory or use `--allow-dirty` for a scientific freeze.

