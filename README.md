# GLM-5.3-Flash EXL3 2.0bpw on one DGX Spark

This repository is the reproducibility and acceptance record for the exact
[`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw)
checkpoint. The accepted baseline runs on one NVIDIA DGX Spark / one GB10 GPU
with CUDA graphs enabled and supports text, structured tools, images, native
video, and a real 200k-token request.

> **Public baseline release verified:** the baseline was accepted
> and then stopped as requested. Hugging Face now exposes all 133 weight shards
> at immutable revision `ab209d2b0a9b822b5caba326c7def8704c97e571`, with the
> full manifest closure verified anonymously. A clean second DGX Spark also
> loaded the exact image and snapshot, captured CUDA graphs, reached a healthy
> API, and passed the five-case behavior replay. The immutable Linux/ARM64 GHCR
> image is public: its index, platform manifest, and config were fetched
> anonymously with matching digests, followed by a pull from an empty Docker
> configuration.

The accepted baseline was usable, but it did not meet the requested 25–50
tok/s target. The measured natural-stop decode estimates were consistently
about 9.3 tok/s. A later DFlash2 Candidate D also passed CUDA-graph, behavior,
vision/video, and fresh 200k-context acceptance. Its sustained 1k-input sample
measured 15.57 tok/s; a short 186-token response after the 200k prefill measured
27.12 tok/s. A subsequent complete 18-request C1 sweep measured 14.00–15.48
tok/s means across 1k–200k inputs, so the sustained 25–50 tok/s target was not
met. See [the full DFlash2 C1 report](BENCHMARKS-D-C1.md).

The separate [manager512/C2 experiment](BENCHMARKS-MANAGER512-C2.md) tests
two active sequences and incoming concurrency up to eight. Behavior, images,
video, and 200k retrieval passed; 23 of 24 initial benchmark cells completed.
The 200k/eight-request cell timed out after six completions, with a separate
retry pending in the report. The sustained 25–50 tok/s target remains unmet.
Its public experimental image and reproduction files are linked in the report;
this configuration has not been promoted to the recommended default or
validated by a clean pull on another node.

The same published weights have also been measured with the **native MTP head
loaded** at the full 262,144-token context, where images, native video, and a
real 262,016-token request all pass and a 1-to-260k sustained sweep shows no
looping at any depth. That is a different runtime image, it is not yet published
to a public registry, and it is documented separately in
[BENCHMARKS-NATIVE-MTP-262K.md](BENCHMARKS-NATIVE-MTP-262K.md). The commands
below run the baseline configuration.

## Exact release facts

| Property | Accepted value |
|---|---|
| Source | `zai-org/GLM-5.3-Flash-BF16` at `a6c167b62691b2bac901344b65cb651a70f53e43` |
| Quantization | Selective EXL3 K2 / 2.0-bpw routed-expert tier |
| Weight shards | 133 root-level safetensors files; 111,352,026,456 bytes |
| Indexed tensors | 583,090 |
| Manifest SHA-256 | `501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3` |
| Runtime | vLLM `0.1.dev20051+g487ecf187`, TP1, DCP1, BF16 activations, FP8 target KV |
| Context | 204,800 configured; 1,638,400 live KV tokens on the final instance |
| CUDA graphs | `FULL_DECODE_ONLY`, capture size 1; graph capture passed |
| Resident model | 89.89 GiB |
| Docker | `ghcr.io/0xsero/glm53-flash-exl3-k2-rankstacked-tp1@sha256:e60a824db7615ead2ae60b4b39b3a9e11e14700bec49901eae7b1e3fb3620d7a` |
| ARM64 manifest | `sha256:c1d0bf2f72da0995eb795d85edb62fe6bd149d740f1dfd8481a1fbd72ebb6a78` |

The `2.0bpw` name describes the routed-expert tier, not every tensor. Routing,
attention, embeddings, vision, and other retained components stay in their
recorded formats. An unused MTP companion layer remains in the checkpoint but
is skipped by the accepted main-model loader.

## Run it

Requirements: one DGX Spark, NVIDIA Container Toolkit, Docker, and the complete
model snapshot on a local filesystem. Do not place credentials in the image or
launch command.

```bash
hf download 0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw \
  --local-dir "$PWD/GLM-5.3-Flash-EXL3-TR3-2.0bpw"

export GLM53_MODEL_ROOT="$PWD/GLM-5.3-Flash-EXL3-TR3-2.0bpw"
export GLM53_IMAGE="ghcr.io/0xsero/glm53-flash-exl3-k2-rankstacked-tp1@sha256:e60a824db7615ead2ae60b4b39b3a9e11e14700bec49901eae7b1e3fb3620d7a"
./runtime/spark/start-rank-stacked-tp1.sh
```

The launcher checks the sealed artifact before starting, mounts weights read
only, refuses to evict an existing CUDA workload, enables chunked prefill, and
keeps CUDA graphs on. The server listens on `127.0.0.1:18080` by default and
serves the ID `glm-5.3-flash-exl3-k2-single-spark`.

Verify the entire public handoff without credentials:

```bash
python3 verify_public_release.py .
```

The verifier checks public/ungated Hub state, exact weight-file closure and LFS
digests, the structural-manifest hash, anonymous GHCR access to both the OCI
index and its Linux/ARM64 manifest, and the pinned public GitHub revision.

Seal the short text, tool-call, and language checks after launch without a
token-cap request field:

```bash
python3 behavior_acceptance.py --output evidence/behavior-final.json
```

For the accepted DFlash2 configuration, first download the external draft at
its pinned revision. It is not bundled because it has a separate
CC-BY-NC-ND-4.0 license.

```bash
hf download IncoAI/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --local-dir "$PWD/GLM-5.3-Flash-DFlash2"

export GLM53_DFLASH_ROOT="$PWD/GLM-5.3-Flash-DFlash2"
export GLM53_IMAGE="ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:6be6de479a5c8c6b8ce9ce42a7be3a2f79e9eeed854ddd4d73e3fc407da88a4d"
./runtime/spark/start-dflash2.sh
```

The Linux/ARM64 manifest, config, and all 57 layer blobs were fetched
anonymously; see `evidence/dflash-docker-publication.json`.

To rebuild the small runtime overlay rather than pulling the release image:

```bash
docker build --platform linux/arm64 \
  -f runtime/spark/Dockerfile.presliced \
  -t local/glm53-flash-exl3-k2-rankstacked-tp1 \
  runtime/spark
```

## What passed

- Exact natural-stop text response and structured tool-call parsing passed a
  fresh clean second-Spark replay; the sanitized payloads are preserved in
  `evidence/behavior-final.json`.
- Four of four paired image fixtures and two of two paired native-video
  fixtures passed again on the clean second Spark; see
  `evidence/vision-clean.json`.
- Arabic, Chinese, and Polish language-constrained smoke prompts passed in the
  same fresh replay.
- A fresh clean-deployment 200,012-token server-reported prompt with four
  random records at approximately 5%, 35%, 65%, and 95%; all four were
  retrieved exactly. It took 507.43 seconds end to end and 475.78 seconds to
  first streamed token; see `evidence/long-context-clean.json`.
- Full-decode CUDA graph capture and health/model-discovery checks.
- DFlash2 Candidate D captured both target and draft full-decode graphs, passed
  the five behavior cases, all four image and both video fixtures, and exact
  four-of-four retrieval from a fresh 200,013-token prompt. See
  `evidence/dflash-accepted.json`.

Preserved sanitized evidence is under [`evidence/`](evidence/). `BENCHMARKS.md`
explains the timing method and its limits. `release.json` distinguishes the
preserved evidence from acceptance observations that still require replay.

## What did not pass or is not claimed

- The 25–50 tok/s target was not reached. Bounded client-side estimates were
  9.16 tok/s at 1k input, 9.47 at 32k, 9.28 at 131k, and a 9.34 tok/s mean for
  two queued 1k requests. `max_num_seqs=1` means incoming concurrency two was
  serialized.
- A simple Python task consumed the full 600-second deadline in reasoning and
  returned no final content. This is a failed responsiveness/code-quality
  probe.
- Exact held-out KLD against the BF16 source is not measured for this exact
  artifact. The pinned BF16 weights alone are 642,652,070,880 bytes, exceeding
  all four local Sparks' combined 512 GiB by 92,896,256,992 bytes before
  runtime, activations, KV cache, or logits. Historical measurements from
  other pruning candidates are not attributed to it.
- The visual fixtures are controlled discrimination checks, not a broad
  multimodal benchmark.
- Earlier DFlash attempts failed target-KV compatibility and then KV admission;
  those failures remain in `evidence/dflash-capacity.json`. Candidate D fixed
  only the drafter manager-block fallback, retained CUDA graphs, admitted
  264,050 KV tokens, and passed the acceptance gates above. Concurrency greater
  than one and a full context/sustained-speed matrix are still unmeasured.
- The post-release exact-K2 runtime abliteration experiment is not an accepted
  abliterated release. Its sealed strength-1 writer projection matched sampled
  materialized BF16 abliterated columns at 99.9978% element agreement and the
  server loaded with CUDA graphs. Ordinary behavior passed 5/5 and images 4/4,
  but all three bounded refusal probes still refused and one of two video cases
  failed to stop. It was stopped and not promoted; see
  `evidence/abliteration-overlay-strength1.json`.

## How this release was produced

### The design decision: spend bytes on bit width, not on deleting experts

The tempting way to fit a 642 GB MoE into one 128 GB Spark is to drop experts.
Measured on one frozen panel, that is a bad trade. Removing experts costs more
fidelity per byte than lowering the bit width of the experts you keep:

| Candidate | Routed-expert policy | Byte cost | BF16 top-1 agreement | Mean KL |
|---|---|---|---|---|
| Q3 reference | all 288 experts at 3-bit | largest | 87.384% | 0.152204 |
| **K2 (this release)** | **all 288 experts at 2-bit** | **111.35 GB** | **77.385%** | **0.438986** |
| K2 keep-256 | 256 of 288 experts at 2-bit | ~95.6 GB | 71.350% | 0.687907 |
| Q3 keep-192 | 192 of 288 experts at 3-bit | ≈ the K2 tier | 61.810% | 1.183637 |
| Q3 keep-176 | 176 of 288 experts at 3-bit | smaller | 58.807% | 1.343988 |

The decisive row pair is *Q3 keep-192* against *K2*. Dropping a third of the
experts **and** paying 3 bits for the survivors still lands at 61.8% agreement
and a mean KL of 1.18 — worse than simply using 2 bits for every expert, at
roughly the same byte cost. Pruning loses twice: it removes the routing options
the model relies on, and the bytes it frees do not buy back the loss.

So the release keeps **every routed expert** and quantizes them uniformly to
EXL3 K2 with the MCG codebook, and reclaims the budget from bit width instead.
This is the whole reason the artifact exists in its current shape.

### What is *not* quantized

Only the routed experts are quantized. Execution-critical and
quality-sensitive tensors stay in their source precision, including attention,
embeddings, the vision tower, the output head, and the MTP head. Those retained
tensors were verified **byte-identical to the separate 3.0bpw release** across
49 shards and 2,482 tensors, totalling 33,835,039,608 bytes. Two artifacts
sharing 33.84 GB of identical bytes is a strong structural check: it proves the
low-bitrate artifact did not silently degrade the components that carry
multimodal and long-context behaviour.

That is also why this checkpoint can be described as vision-preserving rather
than vision-quantized, and why the image, video, and MTP gates above are
meaningful.

### Resulting artifact

| Property | Value |
|---|---|
| Total weight bytes | 111,352,026,456 |
| Retained (unquantized) bytes | 33,835,039,608 |
| Routed-expert bytes | ≈ 77.5 GB, all 288 experts per layer |
| Weight shards | 133 |
| Indexed tensors | 583,090 |
| Resident on one GB10 | 89.89 GiB |
| Reduction vs. BF16 source | ≈ 531 GB, from a 642,652,070,880-byte source |

### The quality gap, and why promotion is held

The panel above is a frozen WikiText2 set of 32 × 2048 tokens — 65,504
next-token positions — scored against the BF16 teacher with a frozen evaluator
identity. It is a *screen*, not an end-to-end acceptance, and it does not
establish broad task quality.

It does establish the gap that gates this release. The 2.0bpw tier reaches
77.385% top-1 agreement against a 3-bit reference at 87.384%, with a mean KL of
0.439 versus 0.152. Against the working bands used for this checkpoint — below
0.01 near-lossless, below 0.05 good, below 0.1 noticeable — a mean KL of 0.439
is well into the noticeable range. This tier is a *serving* baseline, not a
quality-equivalent substitute for the 3-bit reference, and the accepted baseline
is therefore published with an explicit quality caveat rather than as a drop-in
replacement.

An exact held-out KLD against the BF16 source remains unmeasured for this
artifact; the pinned BF16 weights alone exceed the local four-node memory pool
before runtime or logits. See `evidence/kld-feasibility.json`.

### The next direction: non-uniform allocation instead of uniform pruning

If dropping experts is a bad trade, the remaining lever is to spend bits
*unevenly*. Two things were established:

- **Per-layer sensitivity is real.** Raising the bit width of only 14
  down-projections while retaining all 288 experts, chosen using separate
  calibration, measured 78.381% agreement at a cost of 3.94 GiB. That is roughly
  a full point of agreement for a small slice of bytes, but it is still far from
  the 3-bit reference, so it was recorded and not promoted.
- **Per-layer expert counts are runtime-supported.** A modified loader accepts a
  different expert count per layer while keeping the global count at 288. A GPU
  smoke test on two layers with 8 and 16 retained experts passed with exact
  router rows, correctly fused pointer tables, and graph replay matching eager
  execution to 0.0 — meaning non-uniform pruning is not blocked by the runtime.

That combination — uneven bits *and* uneven expert counts, chosen by measured
sensitivity rather than uniformly — is the live research direction. It has no
accepted result, and this repository does not claim one.

## Source and licenses

The model weights follow the source model's MIT license. The runtime overlay is
also MIT. The container inherits third-party packages and their own licenses;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The optional external
DFlash2 draft is accepted at the pinned revision above but is not bundled.
