# SAGE.11 small relational collection result

Status: **COMPLETE — relational fit not yet executed**

Collection date: 2026-07-26

Pre-registration commit: `3ac6d60`

Manifest checksum:
`11a734063ac4be4b8cece50a4d6e7ee40bb25ccfacbc8cd703a1565845f39f2c`

Relational schema checksum:
`84f044dd08f3240f968a6ba1bf528896eab00eb39066dcce95d0f87e9a9193f7`

## Result

The replacement collector completed the frozen 10,027-row target in about
215 seconds with eight independent workers. Every published shard and the
manifest verify.

| Game | Frozen quota | Collected |
| --- | ---: | ---: |
| `bp35` | 1,000 | 1,000 |
| `cd82` | 1,000 | 1,000 |
| `dc22` | 1,000 | 1,000 |
| `g50t` | 1,000 | 1,000 |
| `ka59` | 1,000 | 1,000 |
| `lf52` | 1,000 | 1,000 |
| `lp85` | 27 | 27 |
| `sp80` | 1,000 | 1,000 |
| `su15` | 1,000 | 1,000 |
| `tr87` | 1,000 | 1,000 |
| `tu93` | 1,000 | 1,000 |

No duplicate padding was used for `lp85`.

## Collection policy

The existing outcome-independent policy produced:

- active controller: 7,019 rows;
- uniform legal: 2,005 rows;
- frontier-stall probe: 1,003 rows.

Those counts are the deterministic 70/20/10 schedule over a total that is not
divisible by ten. No outcome was used to select a mixture arm.

## Preserved representation

Every row contains:

- one nested v2 base transition for labels, exact deduplication, streaming
  history, and action-argument auditability;
- one 52-value `sage11-object-relations-v1` pre-action vector;
- its relational schema checksum.

The relation vector contains 22 candidate-independent contact/alignment/
proximity features and 30 current-action-dependent object-relative features.
It contains no raw coordinate, color/value, object identifier, state digest,
outcome, policy arm, or game-identity model column. Raw grids are not
archived. Nested action arguments remain auditable but are converted only to
categorical/topological and object-relative features.

The 11 publishable JSONL shards total approximately 19.7 MB and are tracked
through Git LFS. Resumable base-work shards are local collection scratch and
are intentionally ignored.

## Firewall

Only the 11 registered source-training environments were opened. The
manifest explicitly records:

- source-validation shards opened: false;
- historical shards opened: false;
- holdout shards opened: false.

No source-validation, historical, holdout, or regression-only observation was
used to define, collect, verify, or summarize this corpus.

## Verification

`verify_relational_manifest` reloaded all 10,027 rows, checked every shard
SHA-256 and row count, rejected cross-shard transition duplicates, verified
the split registry and 52-feature schema checksums, and reproduced the
manifest checksum.

Re-running the collection command against a verified complete manifest exits
without reopening environments.

## Next gate

The corpus has not yet been fit. The one allowed empirical step is the frozen
source-train leave-one-game-out relational pilot in
`reports/SAGE11_RELATIONAL_PILOT_PROTOCOL.md`. GPU world-model training
remains prohibited until that pilot passes all four gates.

## Reproduction

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.relational_pilot_collection --workers 8
```

Artifacts:

- `training/sage11/relational_pilot_v1/manifest.json`;
- `training/sage11/relational_pilot_v1/collection_report.json`;
- `training/sage11/relational_pilot_v1/shards/*.jsonl`.
