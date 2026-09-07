# Model card

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
