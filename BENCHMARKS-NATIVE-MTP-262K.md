# Native MTP at full 262,144-token context

This document records what the exact published 2.0bpw checkpoint does when the
**native MTP head is actually loaded**, the request context is raised to the
full native limit, and voice/image/long-context gates are re-run against that
configuration. It is a separate configuration from the baseline recorded in
[`BENCHMARKS.md`](BENCHMARKS.md), not a replacement for it.

The headline result: **vision, native video, native MTP, and a real
262,016-token request all pass on the published weights**, and a sustained
1-to-260k sweep shows no looping or repetition collapse at any depth.

## Identity

| Property | Value |
|---|---|
| Weight manifest SHA-256 | `501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3` |
| Relationship to this release | **identical manifest** to the published 2.0bpw artifact |
| MTP companion | **loaded** (the baseline configuration skips it) |
| Configured context | 262,144 |
| Native `max_position_embeddings` | 1,048,576 |
| Runtime state | `BASELINE_FUNCTIONAL_GATES_PASSED_QUALITY_RELEASE_HOLD` |
| Quality promotion | held — see *What is not claimed* |
| Container image | `sha256:afb74c791853d438810b27bf28994128f5baa2a3bf5c9b50b9ecdad7b9afffd5` |
| Image published publicly | **no** — see *Reproducibility gap* |

Because the manifest hash matches the published artifact, these measurements are
attributable to the same weights that
[`verify_public_release.py`](verify_public_release.py) already checks. They are
*not* the same runtime image, and the difference matters.

## Configuration

| Setting | Value |
|---|---|
| `max_num_seqs` | 1 |
| `max_num_batched_tokens` | 2048 |
| KV capacity admitted | 751,007 tokens |
| `gpu_memory_fraction` | 0.93 |
| Prefix caching | disabled (cold) |
| KV dtype | `fp8_ds_mla` |
| CUDA graphs | `FULL_DECODE_ONLY`, capture size 2 |
| Speculative config | `method: mtp`, `num_speculative_tokens: 1`, `attention_backend: B12X` |
| Resident model | 97.89 GiB |
| Resident KV | 5.84 GiB |
| Sampling | temperature 0, low reasoning effort, `response_format: json_object` |
| Backend identity | Jovian3aada677 + B12x3b862805; EXL3 full K2 target; native MTP routed experts online FP8; target/draft B12X attention; explicit B12x KDA prefill |

## Functional gates

All four gates were re-verified against this configuration on one GB10 with the
controller reporting `runtime_unchanged: true`.

| Gate | Result |
|---|---|
| Real long-context request | **pass** — 262,016 prompt tokens against a 262,144 budget |
| Images | **pass** — 4 of 4 paired fixtures |
| Native video | **pass** — 2 of 2 paired fixtures |
| Native MTP counters | **pass** — advancing counters verified in all 7 depth cells |

The visual fixtures are controlled discrimination checks, not a broad
multimodal benchmark. The long-context gate is a single retrieval request, not a
throughput measurement.

## Sustained sweep, depth 1

Seven input depths, three runs each (warmup plus two measured), incoming
concurrency 1, cold prefix cache, natural stop required. Every timing value was
recomputed from the exact emitted token IDs. The task is a structured
normal-stop JSON answer enumerating the integers 1 through 300, which keeps
every measured decode window above 30 seconds.

| Input tokens | Output tokens | TOTAL decode tok/s (measured mean) | Native prefill tok/s | TTFT s |
|---:|---:|---:|---:|---:|
| 1,023 | 683 | 14.98 | 422–426 | 2.45 |
| 4,095 | 683 | 14.92 | 443–445 | 9.26 |
| 16,383 | 684–691 | 14.86 | 452 | 36.3 |
| 65,535 | 687–691 | 14.81 | 455–457 | 143.8 |
| 131,071 | 682 | 14.81 | 455 | 288.2 |
| 199,999 | 682–683 | 14.83 | 454 | 441.0 |
| 260,095 | 682 | 14.82 | 453 | 574.6 |

Across the 14 measured cells the decode rate spans **14.79–14.98 tok/s** and
native prefill spans **412–456 tok/s**. Decode is essentially flat from 1k to
260k, which is the property that makes the configuration usable at full context:
latency to first token grows with prefill, but steady-state decode does not
degrade.

Per-cell rows, MTP counters, and a SHA-256 for every raw cell are in
[`evidence/native-mtp-sustained-262k.json`](evidence/native-mtp-sustained-262k.json).

## Speculative acceptance

| Measure | Value |
|---|---|
| Draft tokens (14 measured cells) | 7,180 |
| Accepted tokens | 7,166 |
| Acceptance | **99.79%** |
| Per-cell range | 99.13% – 100% |

This is a *task-specific* figure. The stimulus is a deterministic counting task
at temperature 0, which is close to the easiest case a drafter can face. A high
acceptance rate here does not transfer to open-ended generation, and this
document does not claim an MTP-on speedup over any other configuration.

## MTP depth 2

A second depth variant was measured on the same target and runtime at two
depths, two warmups plus two measured repeats each.

| Input tokens | Depth 1 TOTAL decode tok/s | Depth 2 TOTAL decode tok/s | Gain |
|---:|---:|---:|---:|
| 1,023 | 14.978 | 19.114 | **+27.62%** |
| 16,383 | 14.859 | 19.029 | **+28.06%** |

Depth 2 is meaningfully faster, but **it was only admitted at 1,024 and 16,384
input tokens**. There is no depth-2 long-context, vision, or 262k result. The
depth-1 configuration above is the only one with the full gate set, so depth 2
is reported here as a measurement, not as a recommended configuration.

## Loop and repetition acceptance

A looping or repetition-collapsed decoder cannot emit 300 distinct, ordered
values, so the counting task doubles as a degeneracy probe. Every emitted token
ID is retained in the raw cells, so the check runs on exact token streams rather
than re-tokenized text.

| Measure | Depth 1 (21 cells) | Depth 2 (8 cells) |
|---|---|---|
| Cells passed | **21 / 21** | **8 / 8** |
| Longest repeated token n-gram | **2** | **2** |
| Longest run of one repeated token | **1** | **1** |
| Distinct-token ratio | 0.361 – 0.368 | 0.360 – 0.368 |
| Finish reason | `stop` in all cells | `stop` in all cells |
| Max input depth tested | 260,096 | 16,384 |

A longest repeated n-gram of 2 tokens means no phrase of three or more tokens
recurs anywhere in any generation, at any depth, including the 260k cells. There
are no consecutive repeated tokens at all in either configuration.

Run it yourself:

```bash
python3 loop_acceptance.py path/to/cells/*-measured*.json \
  --output evidence/loop-acceptance.json
python3 tests/test_loop_acceptance.py
```

`tests/test_loop_acceptance.py` is the part that matters. It asserts the harness
*rejects* six constructed defects — a short-cycle loop, single-token repetition
collapse, a verbatim repeated block, stalled duplicated values, a length-capped
truncation, and near-constant filler — and *accepts* a correct answer, and
checks the process exit codes. A repetition harness that cannot fail proves
nothing, so the controls ship with it. The controls caught two real defects in
the first version of this harness: an n-gram check that passed when it saturated
its own search cap, and a compression check with its comparison inverted.

Evidence: [`evidence/loop-acceptance-native-mtp-depth1.json`](evidence/loop-acceptance-native-mtp-depth1.json),
[`evidence/loop-acceptance-native-mtp-depth2.json`](evidence/loop-acceptance-native-mtp-depth2.json).

## What is not claimed

- **Quality promotion.** The runtime state is
  `BASELINE_FUNCTIONAL_GATES_PASSED_QUALITY_RELEASE_HOLD`. The published 2.0bpw
  routed-expert tier does not reach the agreement of the 3-bit reference. No
  quality improvement is claimed for the MTP configuration; enabling MTP changes
  speed, not fidelity.
- **Concurrency.** Only `max_num_seqs=1` was admitted. Concurrency 2, 4, and 8
  are explicitly *not admitted*. Queueing clients behind one slot is not a
  concurrency result.
- **A general speed target.** The 25–50 tok/s target from the original brief is
  still unmet at depth 1.
- **Combined near-limit vision.** Images and video passed as separate cases with
  normal media budgets, not at maximum media budget on top of a 262k prompt.
- **Broad capability.** Both the counting task and the visual fixtures are
  synthetic controlled checks.
- **Held-out KLD.** Still unmeasured for this artifact; see
  [`BENCHMARKS.md`](BENCHMARKS.md) and `evidence/kld-feasibility.json`.

## Reproducibility gap

The B12x runtime image that produced these measurements
(`sha256:afb74c791853d438810b27bf28994128f5baa2a3bf5c9b50b9ecdad7b9afffd5`) is
**not yet published to a public registry**, and has not been validated by a pull
from an empty Docker configuration on a second node.

The published baseline and DFlash2 images are unaffected by this: both are
immutable, public, and anonymously verified. The commands in
[`README.md`](README.md) run the *baseline* configuration, whose gates,
limitations, and expected ~9.3 tok/s decode rate are documented in
[`BENCHMARKS.md`](BENCHMARKS.md).

Treat this document as a measured capability report for the current weights, not
as a second one-command recipe. Promoting it requires publishing the image,
re-running the gate set from a clean pull on a second Spark, and recording the
result — the same bar the baseline already cleared.