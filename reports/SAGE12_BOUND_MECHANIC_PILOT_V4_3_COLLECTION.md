# SAGE12 V4.3 source collection

Status: **complete and published before source preflight**

Collection report checksum:
`a842c0bdd99a1e10ad48c03ded447e231a6767e6af7410192b2f21c4b2948722`

## Result

The frozen collector completed all 352 source-training roots:

- 32 roots in each of the 11 SAGE11 source-training games;
- 2,396 replay-verified counterfactual pairs;
- 4,792 executed branch transitions;
- zero replay failures;
- 340 complete depth-three trees;
- 12 naturally truncated trees (`lp85`: 10, `su15`: 2);
- no outcome-adaptive selection or deletion.

Every game except `lp85` and `su15` produced the maximum 224 pairs. `lp85`
produced 164 pairs and `su15` produced 216 because terminal child branches
truncate rather than being replaced after observing an outcome.

## Source capacity before scoring

| Effect | Applicable | Positive | Negative | Frozen minimum per class |
|---|---:|---:|---:|---:|
| target created | 3,893 | 188 | 3,705 | 75 |
| target removed | 2,638 | 619 | 2,019 | 75 |
| target moved | 2,638 | 8 | 2,630 | 75 |

The pair-count gate is satisfied (2,396 versus 2,000). Creation and removal
class capacity are satisfied. Target movement is below the frozen positive
minimum. This table is a raw-label count, not a predictive evaluation; the
source preflight remains responsible for the official fail-closed verdict.
No validation game has been opened.

## Runtime audit

The first invocation used the host Python and failed before creating an
environment because that runtime exposed an incompatible `arc_agi` package.
The second used the registered ARC virtual environment and stopped before
writing a shard because `bp35` exposed duplicate byte-identical legal action
candidates. The replay resolver was patched, tested, documented, committed,
and published before collection. It accepts the first identical candidate;
the restored pre-state hash remains authoritative. No schema, quota, seed,
feature, label, metric, or gate changed.

The successful run used:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage12.bound_mechanic_pilot collect-source
```

The source shards and
`training/sage12/bound_mechanic_pilot_v4_3/source_train_collection_manifest.json`
are immutable prospective source audit data. Publication of this checkpoint
authorizes only the frozen source preflight.
