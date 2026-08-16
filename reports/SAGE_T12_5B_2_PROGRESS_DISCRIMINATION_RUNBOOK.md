# SAGE.T12.5b.2 runbook

Commit the implementation first. Scientific freeze requires a clean worktree.
Then run from the repository root in PowerShell:

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\progress_shadow_t12_5b_r1_bp35"
$Root = ".\training\sage_t\progress_discrimination_t12_5b_2_bp35"

& $Py -m theory.sage_t.causal.progress_discrimination_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\shadow\shadow_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.progress_discrimination_cli audit `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\audit"

& $Py -m theory.sage_t.causal.progress_discrimination_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\audit\discrimination_receipt.json"
```

The audit makes zero SDK or environment calls. It reads the sealed T12.5b-r1
artifacts and writes only compact affordance, contrast, report and receipt JSON.

With the current parent data, the expected diagnostic status is
`FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS`; therefore the `audit`
command is expected to exit with code 3. This is not a CLI integrity error.
Run `status` afterward and inspect:

- `receipt.classification == "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"`;
- `firewall.t12_5b_3_collection_freeze_authorized == true`;
- `firewall.environment_collection_authorized == false`;
- `firewall.t12_5c_control_freeze_authorized == false`.

Do not reuse a non-empty output directory and do not use `--allow-dirty` for a
scientific freeze.

