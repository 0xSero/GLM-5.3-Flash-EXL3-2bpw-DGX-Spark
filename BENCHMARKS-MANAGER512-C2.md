# Manager512: context and concurrency sweep

**The sustained 25–50 decode tok/s target remains unmet.** The initial sweep passed 23 of 24 cells. The 200k-input, eight-request cell completed six requests and hit its 3,600-second client deadline on two; its unmatched native timing is invalid. A separate 7,200-second-deadline retry passed all eight requests with natural stops and matching native counters; the original failed attempt remains below.

One DGX Spark serves the same exact-2-bit target and pinned DFlash2 draft as the main release. The new allocator uses 512-token draft manager blocks, seven speculative tokens, full CUDA graphs at sizes 8 and 16, and at most two active sequences. Incoming concurrency four or eight includes queueing; it is not four or eight simultaneously active sequences.

The server loaded 92.4 GiB of target and draft weights, reported 11.97 GiB available KV memory and a calculated 587,093-token capacity (2.87 full 204,800-token requests). That capacity report is distinct from the configured two-sequence scheduler limit. Five behavior checks, four images, two videos, and exact four-marker retrieval with 200,012 prompt tokens passed before the sweep.

| Input tokens | Incoming requests | Completed | Mean decode estimate (tok/s) | Median first token (s) | Native prefill rate (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1 | 1/1 | 13.94 | 2.65 | 399.7 |
| 1,024 | 2 | 2/2 | 13.11 | 5.19 | 261.1 |
| 1,024 | 4 | 4/4 | 12.19 | 35.17 | 318.8 |
| 1,024 | 8 | 8/8 | 10.78 | 114.02 | 349.4 |
| 8,192 | 1 | 1/1 | 14.06 | 20.01 | 410.8 |
| 8,192 | 2 | 2/2 | 9.79 | 34.24 | 334.2 |
| 8,192 | 4 | 4/4 | 9.22 | 82.41 | 341.7 |
| 8,192 | 8 | 8/8 | 9.51 | 188.68 | 386.7 |
| 32,768 | 1 | 1/1 | 18.36 | 76.98 | 426.3 |
| 32,768 | 2 | 2/2 | 8.02 | 123.46 | 384.0 |
| 32,768 | 4 | 4/4 | 7.52 | 220.92 | 407.4 |
| 32,768 | 8 | 8/8 | 6.40 | 458.19 | 406.0 |
| 65,536 | 1 | 1/1 | 16.24 | 159.78 | 410.6 |
| 65,536 | 2 | 2/2 | 7.20 | 238.78 | 403.1 |
| 65,536 | 4 | 4/4 | 7.28 | 418.07 | 414.0 |
| 65,536 | 8 | 8/8 | 4.53 | 789.86 | 417.0 |
| 131,072 | 1 | 1/1 | 13.74 | 306.34 | 428.3 |
| 131,072 | 2 | 2/2 | 6.94 | 467.14 | 416.6 |
| 131,072 | 4 | 4/4 | 4.50 | 800.50 | 417.4 |
| 131,072 | 8 | 8/8 | 3.35 | 1477.32 | 413.9 |
| 200,000 | 1 | 1/1 | 17.03 | 467.95 | 427.8 |
| 200,000 | 2 | 2/2 | 6.41 | 711.00 | 419.6 |
| 200,000 | 4 | 4/4 | 4.03 | 1206.68 | 416.7 |
| 200,000 | 8 | 6/8 | invalid | invalid | invalid |

Each cell uses one repeat of the same ten-line debugging workload, preserving reasoning and natural response lengths. The previous D report has three repeats at concurrency one; these differing sample counts and generated responses do not establish a controlled speed improvement. All successful cells matched the isolated native timing counters. Failed and successful raw responses are retained in [the evidence directory](evidence/manager512-c2-20260906/).

Decode is a client streaming estimate. Native prefill rate divides prompt-token counter deltas by summed request prefill durations. Concurrent durations overlap; this is not aggregate throughput or GPU-exclusive time. Queueing, interleaved prefill and response-length variation affect the results. Native decode duration is retained separately because speculative first-token accounting differs. No request token cap was used.

## Final-cell retry

The isolated 200k-input, eight-request retry completed **8/8** requests in 4020.00 seconds with natural stops. Mean client decode estimate was **4.17 tok/s**, median first-token latency **2195.94 seconds**, and native prefill rate **417.57 tok/s**. All eight requests matched the native counters. [Raw retry evidence](evidence/manager512-c2-20260906/retry-200k-c8/) is separate from the initial failed cell.

There is now a successful measured result at every requested length/concurrency pair. This is one successful repeat per pair, not the complete multi-repeat release acceptance suite. Sustained speed remains below target. The long latency includes queueing and interleaved prefill with the two-sequence scheduler limit.

## Reproduce

The [experimental build and launch files](experimental/manager512-c2/) change the draft manager fallback from 64 to 512 while retaining divisibility assertions. The target manager block is 7,168 for this seven-token speculation recipe. This patch is not interchangeable with the three-token speculation experiment, whose 6,912-token target block does not tile 512.

Download the exact target using the main README and the separate draft `IncoAI/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`. The draft retains its separate CC-BY-NC-ND-4.0 terms and is not bundled in the image. The vLLM source file retains its original license header. Source, quantization and runtime attribution in the main README also apply here.

```bash
export GLM53_IMAGE=ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:5ea6d04d65d1a2f30f632815776fb765359529ab117818ba2b5e5c5480e34963
docker pull "$GLM53_IMAGE"
# Set GLM53_MODEL_ROOT, GLM53_DRAFT_ROOT and GLM53_CACHE_ROOT to your local directories.
bash experimental/manager512-c2/start.sh
```

Alternatively, rebuild the pinned-parent image with `docker build -t glm53-manager512:local experimental/manager512-c2` and set `GLM53_IMAGE=glm53-manager512:local`.

Run the pinned benchmark on the same host after readiness, with no other API clients:

```bash
docker run --rm --network host --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
  -v "$GLM53_MODEL_ROOT":/model:ro \
  -v "$PWD/evidence/manager512-c2-20260906":/benchmark:ro \
  -v "$PWD":/results \
  "$GLM53_IMAGE" /benchmark/speed_sweep_v2.py \
  --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics \
  --model glm-5.3-flash-exl3-k2-single-spark --tokenizer /model \
  --lengths 1024,8192,32768,65536,131072,200000 \
  --concurrencies 1,2,4,8 --repeats 1 --output-lines 10 --timeout 7200 \
  --output /results/manager512-new-run
```

The example increases the wall deadline to 7,200 seconds to accommodate the eight-request 200k queue. The original run used 3,600 seconds and its failed final cell remains recorded. The measured image config is `sha256:4cc287d0555c9425ab2e483441ebbb2bb6efe984415230e06c5fd97d004d5bf7`; the parent image is pinned in the Dockerfile. A rebuild need not produce the same image config digest.
