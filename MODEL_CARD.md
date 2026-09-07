---
license: mit
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
inference: false
tags:
- glm
- mixture-of-experts
- exl3
---

# GLM-5.3-Flash EXL3 TR3 2.0bpw

The canonical Hugging Face model card is published with the weights at
[`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw).

All 133 weight shards are public and anonymously verified at immutable Hub
revision `ab209d2b0a9b822b5caba326c7def8704c97e571`. The accepted baseline
server was stopped as requested, then the exact public snapshot and image were
loaded cleanly on a second Spark. CUDA graphs, API readiness, text, tools, and
Arabic/Chinese/Polish replay all passed there. The immutable Linux/ARM64 GHCR
image is also public: anonymous index, platform-manifest, and config downloads
matched the pinned digests, and a pull from an empty Docker configuration
passed.

This repository's README contains the same measured runtime scope, limitations,
and attribution. Exact held-out KLD is not available for this artifact: the
pinned BF16 weights alone exceed all four local Sparks' combined 512 GiB before
runtime or logits. The 600-second code-responsiveness probe failed. A corrected
DFlash2 Candidate D later passed CUDA graphs, behavior, all six vision/video
fixtures, and exact retrieval at 200,013 prompt tokens. It measured 15.57 tok/s
on a sustained 1k-input sample and 27.12 tok/s on a short 186-token post-200k
response. A later complete 18-request C1 sweep measured 14.00–15.48 tok/s means
across 1k–200k inputs, so the sustained 25–50 tok/s target was not met. The
[full DFlash2 C1 report](https://github.com/0xSero/GLM-5.3-Flash-EXL3-2bpw-DGX-Spark/blob/main/BENCHMARKS-D-C1.md)
contains the reproducible matrix and per-request evidence. The draft remains an
external pinned CC-BY-NC-ND-4.0 dependency and is not bundled in the image.

The separate
[manager512/C2 experiment](https://github.com/0xSero/GLM-5.3-Flash-EXL3-2bpw-DGX-Spark/blob/main/BENCHMARKS-MANAGER512-C2.md)
tests two active sequences and incoming concurrency up to eight. Behavior,
images, video, and 200k retrieval passed; 23 of 24 initial benchmark cells
completed. The 200k/eight-request cell timed out after six completions, with a
separate retry pending in the report. The sustained 25–50 tok/s target remains
unmet. The report links its public experimental image and reproduction files;
it has not been promoted to the recommended default or validated by a clean
pull on another node.

The post-release strength-1 runtime abliteration experiment was not promoted.
The sealed writer projection loaded with CUDA graphs and matched sampled
materialized BF16 abliterated columns at 99.9978% element agreement. Ordinary
behavior passed 5/5 and images 4/4, but all three bounded refusal probes still
refused and one of two video cases failed to stop. This is a documented negative
result, not an abliterated model release.

The required runtime source and launch recipe are in the public
[DGX Spark runtime repository](https://github.com/0xSero/GLM-5.3-Flash-EXL3-2bpw-DGX-Spark).
The immutable ARM64 image is
`ghcr.io/0xsero/glm53-flash-exl3-k2-rankstacked-tp1@sha256:e60a824db7615ead2ae60b4b39b3a9e11e14700bec49901eae7b1e3fb3620d7a`.
The accepted DFlash runtime image is public at
`ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:6be6de479a5c8c6b8ce9ce42a7be3a2f79e9eeed854ddd4d73e3fc407da88a4d`;
the separately licensed draft checkpoint is still downloaded and mounted at
runtime.

## Release identity

Status: **Public checkpoint**. Audited weight payload: 133 safetensors files, 111,352,026,456 bytes (weight files only; excludes metadata).

Upstream source: [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), BF16 revision `a6c167b62691b2bac901344b65cb651a70f53e43`. Artifact/evidence snapshot inspected: [`2942abd96ee224679bf513d501f6b40dc9237211`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw/tree/2942abd96ee224679bf513d501f6b40dc9237211). A card update does not constitute a new weight conversion.

## Component layout and compatibility

| Component | Storage |
|---|---|
| Routed expert gate/up/down, language layers 3–44 | Selective EXL3 K2 |
| Attention, routers, shared experts, dense layers, embeddings, head, norms and vision | Retained source precision |
| MTP companion | Present; execution is a separate validation gate |

The bitrate labels describe the routed-expert tier, not every tensor. A custom selective-EXL3 loader is required. The accepted single-Spark launch commands, pinned Docker images and checks are in the [runtime README](https://github.com/0xSero/GLM-5.3-Flash-EXL3-2bpw-DGX-Spark).

## Calibration and coverage

600 × 2,048 tokens (1,228,800 tokens): 536 base rows plus 64 scrubbed private-session rows; all 92 protected random rows were retained. Natural top-8 routing was the policy. The [session calibration manifest](evidence/session-calibration-manifest.json) records counts and hashes and labels full-layer route coverage pending; do not infer full coverage from sealing the corpus. Raw private session text is excluded.

## Attribution and license

Z.AI supplies the MIT base model. TurboDerp/ExLlamaV3 supplies EXL3/Trellis. Brandon M. Music is credited for the MIT GLM-5.2 TR3 lineage at `f79c9167690ca705e877ae4dc55a841d1aae1247`; separately licensed GLM-5.3 artifacts are not the source of this release. The independent conversion workflow is Dione. vLLM and the runtime contributors are credited with their licenses in the [third-party notices](https://github.com/0xSero/GLM-5.3-Flash-EXL3-2bpw-DGX-Spark/blob/main/THIRD_PARTY_NOTICES.md). The optional IncoAI DFlash2 draft retains its separate CC-BY-NC-ND-4.0 license and is not bundled.

## Intended use and limitations

Use populated checkpoints for local inference or quantization research with the declared compatible runtime. Results from one bitrate or runtime do not transfer automatically to another. Quantization may change behavior and factual accuracy; controlled smoke tests do not establish broad benchmark quality. A projection or REAP observation record alone does not prove successful refusal removal or a pruned model release.

## Related releases

| Repository | Access | Weight files | Weight payload (GB) |
|---|---|---:|---:|
| [EXL3-Q4](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4) | public | 217 | 187.453 |
| [EXL3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw) | public | 130 | 149.403 |
| [EXL3-TR3-2.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw) | public | 133 | 111.352 |
| [EXL3-2.5bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-2.5bpw) | public; no weights | 0 | 0.000 |
| [EXL3-2.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-2.0bpw) | public; no weights | 0 | 0.000 |
| [EXL3-TR3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-3.0bpw) | public; no weights | 0 | 0.000 |
| [EXL3](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3) | public; index only | 0 | 0.000 |
| [BF16-Abliterated](https://huggingface.co/0xSero/GLM-5.3-Flash-BF16-Abliterated) | private | 120 | 642.652 |
| [Abliterated-EXL3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-Abliterated-EXL3-3.0bpw) | private | 133 | 149.403 |
| [Abliterated-EXL3-Q4](https://huggingface.co/0xSero/GLM-5.3-Flash-Abliterated-EXL3-Q4) | private | 133 | 187.454 |
| [Abliterated-EXL3](https://huggingface.co/0xSero/GLM-5.3-Flash-Abliterated-EXL3) | private; index only | 0 | 0.000 |

Private links require authorized access. Two suite indexes and three placeholders are included in this inventory; they are not additional trained models.

## Evidence files

- [EXL3_MANIFEST.json](EXL3_MANIFEST.json)

## REAP observation provenance

The original 3bpw and Q4 were each observed on two corpora: private calibration material and balanced 12-language Wikipedia. Each sealed lane records 128 sequences × 1,024 tokens (131,072 tokens), across 42 routed layers and 288 experts per layer. These four observation lanes are separate from quantization calibration and do not mean experts have been pruned from the weights above. The [private observation dataset](https://huggingface.co/datasets/0xSero/glm53-flash-exl3-reap-observations) holds the manifests and aggregate sidecars.
