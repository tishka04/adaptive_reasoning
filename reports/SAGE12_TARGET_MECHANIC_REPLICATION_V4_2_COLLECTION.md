# SAGE12 target-mechanic V4.2 — prospective collection

Date: 2026-07-27

Status: `COMPLETE`

Collection report checksum:
`6bdec774c744061e3e5014ced8d3d0191d1cdc13243130817ea9ec84fd50dce7`

Combined shard checksum:
`8aa57cd83fa93ba08a7bc309a62f3e68509778628e6c588d4b9ee8b3bbe4534b`

The source preflight passed and authorized exactly 768 new transitions.
Collection completed with 256 rows per frozen source-validation game, eight
resets per game, at most 32 actions per reset, and policy seeds 479, 523,
569, and 617. Selection was outcome-independent and chronological repeats
were retained.

| Game | Rows | Action coverage | Exact repeats retained | Created | Removed | Moved |
|---|---:|---|---:|---:|---:|---:|
| `re86` | 256 | 51–52 across 5 actions | 6 | 0 | 0 | 205 |
| `ls20` | 256 | 64 across 4 actions | 71 | 0 | 0 | 5 |
| `sc25` | 256 | 51–52 across 5 actions | 14 | 50 | 50 | 4 |
| **Total** | **768** | — | **91** | **50** | **50** | **214** |

Shard checksums:

- `re86`: `d70974ed868a0929bd73217d2854d10775acae38658f255aa827fbf4df2d7c9d`;
- `ls20`: `1e95071943e585dc8635548e69881d4f9d43ab85a40bd1188c0a938c9870e11d`;
- `sc25`: `711402b70870ad058017443564436c3ad47dde7d7661dc7461df5d40bff90fbb`.

The transition-level counts are an audit of the frozen shards, not the final
window-level gate result. No model metric, baseline comparison, Qwen output,
or V5 decision had been computed when this checkpoint was published.
Holdout, historical, and `ar25` data remain unopened.
