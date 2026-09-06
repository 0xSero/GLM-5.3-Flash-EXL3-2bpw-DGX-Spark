# DFlash2 Candidate D: complete C1 context sweep

All 18 measured requests completed with natural `stop`, with three repeats at
each of six input lengths. Every cell matched the isolated server timing counters.
The sustained 25–50 decode tok/s target was **not met** on this workload.

| Target input tokens | Mean decode estimate (tok/s) | Median first-token latency (s) | Mean native prefill rate (tok/s) |
|---:|---:|---:|---:|
| 1,024 | 14.74 | 2.63 | 394.9 |
| 8,192 | 15.14 | 20.12 | 403.6 |
| 32,768 | 14.25 | 77.33 | 413.7 |
| 65,536 | 14.00 | 152.86 | 422.9 |
| 131,072 | 14.81 | 305.36 | 426.9 |
| 200,000 | 15.48 | 466.42 | 429.2 |

This is one DGX Spark / one GB10 GPU, incoming concurrency 1, with the
204,800-context DFlash2 recipe in [start-dflash2.sh](runtime/spark/start-dflash2.sh).
The target and draft both captured full CUDA graphs. The restored instance passed
all five behavior checks before the sweep. The configuration uses seven speculative
tokens, graph size 8, FP8 cache, and one active sequence. This report does not
establish concurrent execution or the speed of other workloads.

The benchmark requests ten numbered debugging facts and retains the model's
reasoning behavior. Each cell uses a fresh padded prompt. All responses, including
reasoning, are saved in [the evidence directory](evidence/dflash-d-c1-20260906/).
No request token cap is used. A client wall deadline of 1,800 seconds applies;
none of the 18 measured cells timed out. The warmup is excluded from the table.
The earlier 27.12 tok/s result was a shorter 186-token retrieval response and should
not be substituted for this sustained workload.

Decode values are client streaming estimates. Native prefill rates divide the
server's prompt-token counter delta by its request prefill-time delta. First-token
latency also includes request processing and first decode work. Server decode
durations are preserved separately; speculative first-token accounting prevents
assuming those durations have the same denominator as the client estimate.

## Exact artifacts

- Target: [0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw), with the weight manifest pinned in the main release.
- External draft: `IncoAI/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`.
- Public image: `ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:6be6de479a5c8c6b8ce9ce42a7be3a2f79e9eeed854ddd4d73e3fc407da88a4d`.
- Image config: `sha256:3d2e379b1bd8b18c5ecdbd6a1fba0b80f2964c94ecc642a69b9432f966727a36`.
- Benchmark SHA-256: `ecffdc551594a12fcaf6ca9985d7cbc46b33a89a06dadfed50602d168e9452f9`.
- Run started: `2026-09-06T06:22:31.488640+00:00`.

## Repeat the sweep

Start the pinned DFlash2 recipe from the README, then run the benchmark on the
same host with an otherwise idle API. The benchmark container uses CPU-only
client code; the serving container owns the GPU. Set `GLM53_MODEL_ROOT` and
`GLM53_IMAGE` as described in the README. The command below uses the launcher's
default endpoint and served name.

```bash
docker run --rm --network host --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
  -v "$GLM53_MODEL_ROOT":/model:ro \
  -v "$PWD/evidence/dflash-d-c1-20260906":/benchmark:ro \
  -v "$PWD":/results \
  "$GLM53_IMAGE" /benchmark/speed_sweep_v2.py \
  --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics \
  --model glm-5.3-flash-exl3-k2-dflash --tokenizer /model \
  --lengths 1024,8192,32768,65536,131072,200000 \
  --concurrencies 1 --repeats 3 --output-lines 10 --timeout 1800 \
  --output /results/dflash-c1-new-run
```

The run metadata records the exact script hashes. Fresh prompt identifiers and
natural response lengths vary, so repeat the matrix rather than expecting identical
token counts or timing. Source and runtime attribution and the external draft's
separate license remain documented in the main README.
