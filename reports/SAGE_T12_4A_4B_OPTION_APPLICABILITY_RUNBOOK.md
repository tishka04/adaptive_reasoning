# SAGE.T12.4a.4b runbook

Run from the repository root in PowerShell after committing the implementation.
The freeze must see a clean worktree.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\option_transfer_t12_4a_4_bp35"
$Root = ".\training\sage_t\option_applicability_t12_4a_4b_bp35"

& $Py -m theory.sage_t.causal.option_applicability_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\transfer\transfer_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.option_applicability_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\audit" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.option_applicability_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\audit\applicability_receipt.json"
```

Expected successful audit status:
`PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE`. This means the negative transfer was
reproduced and classified; it does not mean the option transferred.

Do not rerun into an existing audit directory. Each run is immutable. On a
failure, inspect `applicability_report.json` and `applicability_diagnosis.json`
before proposing a child experiment. Do not open source validation, holdout,
active control, neural training or production authority.

