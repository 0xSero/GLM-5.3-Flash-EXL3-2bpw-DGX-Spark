# GLM-5.3-Flash EXL3 2.0bpw on one DGX Spark

This repository is the reproducibility and acceptance record for the exact
[`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw)
checkpoint. The accepted baseline runs on one NVIDIA DGX Spark / one GB10 GPU
with CUDA graphs enabled and supports text, structured tools, images, native
video, and a real 200k-token request.

The model is usable, but it did not meet the requested 25–50 tok/s target. The
measured natural-stop decode estimates were consistently about 9.3 tok/s.

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

After `release.json` contains the immutable Hugging Face revision, verify the
entire public handoff without credentials:

```bash
python3 verify_public_release.py .
```

The verifier checks public/ungated Hub state, exact weight-file closure and LFS
digests, the structural-manifest hash, anonymous GHCR access to both the OCI
index and its Linux/ARM64 manifest, and the pinned public GitHub revision. It
fails closed while any publication gate is still pending.

Seal the short text, tool-call, and language checks after launch without a
token-cap request field:

```bash
python3 behavior_acceptance.py --output evidence/behavior-final.json
```

To rebuild the small runtime overlay rather than pulling the release image:

```bash
docker build --platform linux/arm64 \
  -f runtime/spark/Dockerfile.presliced \
  -t local/glm53-flash-exl3-k2-rankstacked-tp1 \
  runtime/spark
```

## What passed

- Exact natural-stop text response and structured tool-call parsing.
- Four of four paired image fixtures and two of two paired native-video
  fixtures.
- Arabic, Chinese, and Polish language-constrained smoke prompts.
- A fresh 200,012-token server-reported prompt with four random records at
  approximately 5%, 35%, 65%, and 95%; all four were retrieved exactly. It
  took 498.40 seconds end to end and 474.04 seconds to first streamed token.
- Full-decode CUDA graph capture and health/model-discovery checks.

Sanitized raw evidence is under [`evidence/`](evidence/). `BENCHMARKS.md`
explains the timing method and its limits.

## What did not pass or is not claimed

- The 25–50 tok/s target was not reached. Bounded client-side estimates were
  9.16 tok/s at 1k input, 9.47 at 32k, 9.28 at 131k, and a 9.34 tok/s mean for
  two queued 1k requests. `max_num_seqs=1` means incoming concurrency two was
  serialized.
- A simple Python task consumed the full 600-second deadline in reasoning and
  returned no final content. This is a failed responsiveness/code-quality
  probe.
- Exact held-out KLD against the BF16 source is not measured for this exact
  artifact. Historical measurements from other pruning candidates are not
  attributed to it.
- The visual fixtures are controlled discrimination checks, not a broad
  multimodal benchmark.
- DFlash2 was tested and rejected from this release because its draft metadata
  inherited the target MLA KV format and failed initialization. Context was
  not reduced and eager mode was not used as a workaround.

## Source and licenses

The model weights follow the source model's MIT license. The runtime overlay is
also MIT. The container inherits third-party packages and their own licenses;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The optional external
DFlash2 draft is not bundled.
