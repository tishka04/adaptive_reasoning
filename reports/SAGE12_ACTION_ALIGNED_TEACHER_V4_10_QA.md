# SAGE12 V4.10 — Action-aligned teacher QA

Status: **TEACHER READY**. This checkpoint was produced and published before
student fitting.

## Corpus

- V4.9 base records: 7,541.
- Fresh V4.10 records: 1,587.
- Combined records: 9,128.
- Same-prestate pairs: 2,396.
- Fresh duplicates against V4.9: zero (the collector rejected them earlier).
- Action-aligned graphs passing the compass/firewall check: 100%.
- Source validation, holdout and online/live environments opened: no.

The capacity amendment checksum
`cef75fd72e8fd2f7b89673b330870301d59e718ef9b64dfaa2ddf9104106f0e9`
is bound into the QA.

## Capacity after augmentation

| Effect | Positive | Games with ≥1 |
|---|---:|---:|
| changed | 8,565 | 11 |
| moved | 1,944 | 10 |
| target_created | 374 | 4 |
| target_removed | 1,023 | 4 |
| target_moved | 183 | 6 |
| level_complete | 6 | 2 |
| game_over | 179 | 11 |
| local_change | 2,144 | 11 |
| path_opened | 404 | 2 |
| path_closed | 130 | 2 |
| actor_approached_root | 62 | 5 |
| contact_gained | 36 | 2 |
| contact_lost | 1,598 | 8 |
| reachable_area_increased | 179 | 5 |
| reachable_area_decreased | 209 | 6 |
| productive | 1,440 | 9 |
| risk | 461 | 11 |

The new collection broadens common functional supervision and adds one genuine
completion. It cannot make mechanic-specific effects portable: completion,
path change and contact gain still occur in only two games. Those facts remain
part of the frozen LOGO result rather than being hidden through resampling.

## Content addresses

- Teacher corpus:
  `78d2e51cf86d979006d7752955040c7b6775989624014653497aa990523c994a`
- Same-prestate pairs:
  `53fa00b17581f22c28fa58f79d97308f1150da935154c0f163c4025cc318c751`
- Teacher QA:
  `37a63f2a45ca53a703c2b17f61aaad88570a403a270e6ce36e7fe9b0aba35bc0`

The next authorized step is the frozen 11-fold GPU evaluation.
