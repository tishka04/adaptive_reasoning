# SAGE12 V4.2.1 prospective collection result

Date executed: 2026-07-27

Status: `COMPLETE`

Collection report checksum:
`8fc7989a30ec4a42e2c1d9d8f49592dc37371d17d1c9f0f406770c62e3fb8785`

Combined shard checksum:
`9cbc1dcb450a71f1a670e515b5adcd7d72af7d9c9fd21549a9b1514917d65a4c`

## Result

The source-authorized collector produced exactly 768 new chronological
transitions, 256 per frozen source-validation game. It used the V4.2.1 seeds
661, 709, 757, and 809 and did not reuse any V4.2 shard.

| Game | Rows | Resets | Retained exact repeats |
| --- | ---: | ---: | ---: |
| `re86` | 256 | 8 | 5 |
| `ls20` | 256 | 8 | 58 |
| `sc25` | 256 | 8 | 16 |
| **Total** | **768** | **24** | **79** |

Legal actions were balanced within one count in every game:

- `re86`: actions 1-5 received 51-52 rows each;
- `ls20`: actions 1-4 received 64 rows each;
- `sc25`: actions 1, 2, 3, 4, and 6 received 51-52 rows each.

Selection was not outcome-adaptive. Chronological repeats were retained as
mechanic evidence. Holdout, historical, and `ar25` data remained closed.

The three Git-LFS shards have independent SHA-256 checksums:

- `re86`: `960300e67e134d130f3a7fa5e9116dd1350946e0bd54ba2b563207e52338d5e3`;
- `ls20`: `c3f5fc299e3236857009d6d9d7c7a5b3c7e7ccc0ff30226971924844fe9b3471`;
- `sc25`: `4a476c06d080bc86e0aaa0e55b0afc248ad04c20998a747d08d3a80a68b00386`.

## Boundary and authority

No prospective window, prediction, metric, control, or Qwen output was
computed before this publication checkpoint. The raw collection now
authorizes the single frozen V4.2.1 evaluation. It does not authorize V5,
world-model fitting, EBM fitting, or controller use.
