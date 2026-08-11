# SAGE.T11 causal experiment CLI runbook

## Scientific boundary

The CLI runs only preregistered source-train, source-validation, historical,
or regression experiments. It never opens the neural holdout or production
authority. A clean Git tree is required for a scientific manifest. The
`--allow-dirty` option creates a smoke-only manifest whose gates cannot pass.

Every experiment binds:

- the causal protocol checksum;
- the Git commit and clean/dirty state;
- hashes of the causal runtime, unified controller, and live runner;
- the sealed complete-program registry;
- the sealed exact-prefix intervention plan;
- games, seeds, resets, action budget, arms, and authority mode;
- an optional passing parent receipt for source validation.

Artifacts are immutable first-writer files. Repeating a phase requires a new
output directory rather than overwriting an existing receipt.

## 1. Raw complete-program registry

Prepare a JSON file with at least two structurally distinct complete programs
per game. Every rival must declare the same complete action catalog so all
particles can predict every candidate intervention.

```json
{
  "games": {
    "bp35": {
      "action_catalog": ["ACTION1", "ACTION2", "ACTION3"],
      "programs": [
        {"format_version": "sage-t-causal-program-v1", "program_id": "..."},
        {"format_version": "sage-t-causal-program-v1", "program_id": "..."}
      ]
    }
  }
}
```

The omitted fields are the complete `CausalProgram.to_dict()` payload. Seal it:

```powershell
& $Py -m theory.sage_t.causal.experiment_cli seal-programs `
  --input .\training\sage_t\causal_inputs\programs.raw.json `
  --output .\training\sage_t\causal_inputs\programs.sealed.json
```

The command compiles every program and rejects missing parents, unresolved
mechanisms, duplicate programs, incomplete action catalogs, and invalid games.

## 2. Raw intervention plan

Each bundle needs a previously measured exact prefix hash, a non-empty prefix,
and at least two distinct legal branches. Branch outcomes are absent: they must
not be observed before the plan is sealed.

```json
{
  "bundles": [
    {
      "bundle_id": "bp35-frontier-01",
      "game_id": "bp35",
      "prefix_hash": "<exact state and action-schema signature>",
      "prefix": [
        {"action_name": "ACTION2", "action_data": {}}
      ],
      "branches": [
        {"action_name": "ACTION1", "action_data": {}},
        {"action_name": "ACTION3", "action_data": {}}
      ]
    }
  ]
}
```

Seal it against the program registry:

```powershell
& $Py -m theory.sage_t.causal.experiment_cli seal-bundles `
  --input .\training\sage_t\causal_inputs\bundles.raw.json `
  --program-registry .\training\sage_t\causal_inputs\programs.sealed.json `
  --output .\training\sage_t\causal_inputs\bundles.sealed.json
```

## 3. Freeze the paired source-train pilot

Commit the implementation and sealed inputs first. Then freeze the matrix:

```powershell
& $Py -m theory.sage_t.causal.experiment_cli freeze `
  --program-registry .\training\sage_t\causal_inputs\programs.sealed.json `
  --bundle-plan .\training\sage_t\causal_inputs\bundles.sealed.json `
  --manifest .\training\sage_t\causal_bp35_v1\manifest.json `
  --stage source_train `
  --games bp35 `
  --seeds 0,1,2 `
  --resets 3 `
  --action-budget 64 `
  --authority bounded
```

Default paired arms are:

- `baseline`;
- `posterior_full`;
- `no_posterior_update`;
- `no_information_gain`;
- `no_a40_memory`;
- `no_mdl_prior`.

Optional sealed arms are `no_intergame_mechanisms` and `symbolic_only`.

## 4. Execute exact-prefix bundles

All particle predictions are computed and registered before the first branch is
executed. Each branch reconstructs a fresh environment, replays the same
prefix, verifies the state/action-schema hash, executes one legal action, and
updates the fixed preregistered rival set. Repairs are disabled in this phase
so measured entropy change is not confounded by hypothesis-set expansion.

```powershell
& $Py -m theory.sage_t.causal.experiment_cli replay `
  --manifest .\training\sage_t\causal_bp35_v1\manifest.json `
  --output-dir .\training\sage_t\causal_bp35_v1\replay `
  --environments-dir .\environment_files
```

Bounded authority remains closed unless `replay_receipt.json` reports
`PASS_CAUSAL_REPLAY_GATE` with exact hashes, at least two branches per bundle,
pre-execution predictions, and positive total entropy reduction.

## 5. Execute paired baseline/full/ablation arms

```powershell
& $Py -m theory.sage_t.causal.experiment_cli run `
  --manifest .\training\sage_t\causal_bp35_v1\manifest.json `
  --replay-receipt .\training\sage_t\causal_bp35_v1\replay\replay_receipt.json `
  --output-dir .\training\sage_t\causal_bp35_v1\paired `
  --environments-dir .\environment_files
```

Every arm receives the same game, seed, reset index, initial visual state, and
action budget. A fresh controller is constructed at every reset. Full/A40
arms reload only their own game-seed-arm memory; `no_a40_memory` starts from
the sealed rivals. Memory paths include the game id, preventing evidence from
one validation game from initializing another.

The source-train scientific gate additionally requires real progress, zero
safety regressions, and an advantage over `no_posterior_update`. Failure writes
a signed negative receipt and opens nothing.

## 6. Verify all bindings

```powershell
& $Py -m theory.sage_t.causal.experiment_cli status `
  --manifest .\training\sage_t\causal_bp35_v1\manifest.json `
  --replay-receipt .\training\sage_t\causal_bp35_v1\replay\replay_receipt.json `
  --gate-receipt .\training\sage_t\causal_bp35_v1\paired\gate_receipt.json
```

## 7. Frozen source validation

Only a passing source-train receipt may be bound into a source-validation
manifest:

```powershell
& $Py -m theory.sage_t.causal.experiment_cli freeze `
  --program-registry .\training\sage_t\causal_inputs\validation_programs.sealed.json `
  --bundle-plan .\training\sage_t\causal_inputs\validation_bundles.sealed.json `
  --manifest .\training\sage_t\causal_validation_v1\manifest.json `
  --stage source_validation `
  --games re86,ls20,sc25 `
  --seeds 2101,2102,2103,2104,2105 `
  --resets 14 `
  --action-budget 96 `
  --authority bounded `
  --parent-receipt .\training\sage_t\causal_bp35_v1\paired\gate_receipt.json
```

The validation gate requires progress in at least two games, zero safety
regressions, and an advantage over the posterior-update ablation. Even a pass
does not make the holdout accessible through this CLI.

