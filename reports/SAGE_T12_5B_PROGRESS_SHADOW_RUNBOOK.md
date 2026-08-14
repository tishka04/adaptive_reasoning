# SAGE.T12.5b runbook

Commit the T12.5b implementation first: the scientific freeze requires a clean
worktree. Then run from the repository root in PowerShell.

```powershell
$Py = (Resolve-Path .\ARC-AGI-3-Agents\.venv\Scripts\python.exe).Path
$Parent = ".\training\sage_t\causal_progress_t12_5_bp35"
$Root = ".\training\sage_t\progress_shadow_t12_5b_r1_bp35"

& $Py -m theory.sage_t.causal.progress_shadow_cli freeze `
  --parent-manifest "$Parent\manifest.json" `
  --parent-receipt "$Parent\compiled\causal_progress_receipt.json" `
  --manifest "$Root\manifest.json"

& $Py -m theory.sage_t.causal.progress_shadow_cli run `
  --manifest "$Root\manifest.json" `
  --output-dir "$Root\shadow" `
  --environments-dir .\environment_files

& $Py -m theory.sage_t.causal.progress_shadow_cli status `
  --manifest "$Root\manifest.json" `
  --receipt "$Root\shadow\shadow_receipt.json"
```

The expected pass status is `PASS_T12_5B_PROGRESS_SHADOW_GATE`. A pass requires
all entries under `metrics.checks` to be true. In particular, inspect:

- `exact_prefix_rate == 1`;
- `branch_availability_rate == 1`;
- `effect_determinism_rate == 1`;
- `effect_transport.milestone_signature_transport_rate == 1`;
- `ranking.per_method.causal_progress.top1_accuracy == 1`;
- `ranking.per_method.causal_progress.mean_reciprocal_rank == 1`;
- `observed_confirmation.top1_accuracy == 1`;
- `terminal_failures == 0`;
- `sdk_calls.within_budget == true`;
- `storage.within_budget == true`.

`effect_transport.exact_projection_transport_rate` is diagnostic and may be
below one because non-contract aggregate relations differ across the two sealed
contexts. Do not change the protocol after seeing the new run.

The command executes the amended fixed 60-branch source-train collection.
`ACTION7` is explicitly excluded because it is advertised by the frame but is
not executable through the SDK at any sealed stage. Rankings do not influence
those actions. Do not reuse a non-empty output directory and do not use
`--allow-dirty` for a scientific freeze.
