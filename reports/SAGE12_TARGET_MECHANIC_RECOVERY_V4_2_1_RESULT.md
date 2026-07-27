# SAGE12 V4.2.1 target-mechanic recovery result

Date executed: 2026-07-27

Overall status: `FAIL_CLOSED`

Result checksum:
`27861c650c1cd51f5ee96c03e3ae297497a4d04e39f49391b1631840b43757ff`

Structured checksum:
`b6cb09906e5709b129f44772a0ffd429b3a42aa732bc09f3aca71f91ffef2b72`

## Decision

V4.2.1 repaired the V4.2 runtime defect and produced a complete,
transactional prospective verdict. The structured target-mechanic inducer
passed 18 of 19 gates but missed the frozen anchor-binding-shuffle threshold.
Its Brier-skill loss under that control was +0.017061 against the required
+0.020000, a shortfall of 0.002939. Because the protocol is conjunctive,
V4.2.1 is `FAIL_CLOSED`.

Qwen failed its separate branch. No V5 protocol, semantic world model, EBM,
shadow controller, bounded probe, or active controller is authorized.

## Structured result

The evaluator derived 576 unique prospective windows from the 768 newly
published transitions and wrote all 576 predictions before Qwen. The public
serializer successfully encoded 228 selected generic-`any` evidence rules.
Prediction SHA-256:
`c44bd5d89b451802d060509b2c2f297461598616ea2a972cdaf44d4e0a230e37`.

Against the stronger local-action baseline:

| Metric | Structured result |
| --- | ---: |
| Calibrated macro Brier | 0.032244 |
| Calibrated macro Brier skill | +0.703788 |
| Raw macro Brier skill | +0.442526 |
| Calibrated macro-F1 | 0.937460 |
| Calibrated macro-F1 gain | +0.307529 |
| Macro-ECE | 0.071950 |
| Bootstrap skill 95% interval | [+0.616474, +0.753811] |
| Context Brier skill | +0.761707 |
| Outcome-shuffle skill loss | +0.462532 |
| Binding-shuffle skill loss | **+0.017061** |

Every validation game transferred positively:

| Game | Brier skill |
| --- | ---: |
| `re86` | +0.792594 |
| `ls20` | +0.540818 |
| `sc25` | +0.137379 |

All three target effects individually met their capacity, Brier, and F1
eligibility criteria. Strict JSON rendering, grounding, `support=0`,
prospective capacity, raw/calibrated utility, bootstrap, macro-F1, outcome
shuffle, context gain, per-game transfer, calibration, identity leakage,
source preflight, source rehearsal, actor exclusion, and model-view firewall
all passed. Only the preregistered binding-shuffle gate failed.

## Interpretation

This is strong evidence that short semantic transition histories predict
target creation, removal, and movement across the three validation games.
The result is substantially stronger than action identity alone and is well
calibrated at the aggregate level.

The failed control matters, however. Shuffling the concrete action-anchor
binding preserved nearly all predictive skill. The structured model is using
temporal effect regularities very effectively, but the experiment did not
prove that it learned enough object/action binding to support grounded
counterfactual rollouts. The result therefore supports temporal mechanic
induction while withholding the stronger claim needed for a semantic world
model.

The narrow miss should not be repaired by lowering the threshold after
inspection. A successor must be a new protocol that changes the experiment,
for example by testing matched counterfactual bindings with sufficient
within-context variation. V4.2.1 itself remains a closed negative promotion
result.

## Separate Qwen result

Qwen2.5-0.5B-Instruct evaluated 128 clean contexts and 128
outcome-shuffled controls on `cuda:0`. It required 1,093.33 seconds of model
inference and failed all six separate gates:

- strict JSON validity, grounding, `support=0`, and emitted hypotheses were
  all zero;
- productive-effect recall@8 was zero;
- outcome-shuffle skill loss was zero;
- Brier skill versus local action was -0.295586;
- `ls20` and `sc25` transferred negatively.

All 256 raw responses were Markdown-fenced JSON, so the strict parser rejected
them at the first character. The first inspected payload also used effect
codes such as `M` as action-scope values, which would not satisfy exact query
grounding even after removing fences. No post-hoc fence stripping or prompt
repair was allowed.

## Transaction and operational audit

The first invocation reached the external six-minute command limit while
Qwen was still running. This was not an exception raised by the pilot. By
then, `validation_windows.jsonl`, all predictions, and
`structured_intermediate.json` had already been written. The persisted
structured checksum was
`b6cb09906e5709b129f44772a0ffd429b3a42aa732bc09f3aca71f91ffef2b72`.

The exact frozen command was retried once with only a longer orchestration
timeout. No code, data, seed, model, prompt, schema, decoding, metric,
threshold, or gate changed. It completed in 1,116.9 seconds and reproduced
the identical structured checksum before writing both Qwen streams and the
final result. This confirms that the transactional ordering protected the
structured verdict from a later operational interruption.

## Authority

- V5 protocol: not authorized;
- Qwen in V5: not authorized;
- semantic world-model fitting: not authorized;
- EBM fitting: not authorized;
- controller evaluation or activation: not authorized;
- holdout, historical, and `ar25`: not opened.

V4.2.1 artifacts remain immutable audit evidence. Any successor must receive
a new version, frozen manifest, prospective data policy, and protocol before
new outcomes are opened.

Final artifact validation passed: both JSON checksums re-derived exactly,
the four JSONL streams contained 576 validation windows, 576 structured
predictions, 128 clean Qwen rows, and 128 shuffled Qwen rows, and the 26
targeted V4.2/V4.2.1 tests plus focused Ruff checks passed.
