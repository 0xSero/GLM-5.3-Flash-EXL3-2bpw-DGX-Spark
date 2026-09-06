# Acceptance ledger

The exact model/artifact identity and final runtime configuration are fixed for
every pass below. Changing the weights, chat template, image, KV mode, context,
or graph configuration invalidates the affected gates.

| Gate | Status | Evidence or qualification |
|---|---|---|
| Structural weight/index closure | pass | 133 shards, 583,090 tensors, manifest `501641b…8ba3` |
| Sanitized publication tree | pass | 111,352,026,456 content-addressed bytes; no symlinks, forbidden names, credentials, or private paths |
| One GB10 GPU / TP1 | pass | final live runtime |
| CUDA graph capture | pass | `FULL_DECODE_ONLY`, size 1 |
| Text natural stop | pass | clean second-Spark replay in `evidence/behavior-final.json` |
| Structured tool call | pass | clean replay parsed exact function name and JSON arguments |
| Multilingual smoke | pass | clean replay passed Arabic, Chinese, and Polish |
| Images | pass | clean second-Spark replay 4/4 in `evidence/vision-clean.json` |
| Native video | pass | clean second-Spark replay 2/2 in `evidence/vision-clean.json` |
| Real 200k request | pass | clean replay: 200,012 server prompt tokens, unique nonce, 4/4 exact retrieval, natural stop in `evidence/long-context-clean.json` |
| 25–50 tok/s target | **fail** | bounded estimates cluster near 9.3 tok/s |
| Code responsiveness | **fail** | simple Python request had no final content by 600 seconds |
| Exact held-out KLD | unmeasured | pinned BF16 weights are 642.65 GB, exceeding the four-Spark 512 GiB pool before runtime/logits; see `evidence/kld-feasibility.json` |
| DFlash2 Candidate D | pass at max sequences 1 | target and draft CUDA graphs, behavior 5/5, images 4/4, videos 2/2, fresh 200,013-token exact retrieval 4/4; see `evidence/dflash-accepted.json` |
| DFlash2 sustained speed | below target / partial | 15.57 tok/s at 1k input; a short 186-token post-200k response measured 27.12 tok/s, but no full sweep or concurrency result exists |
| Public HF weights | pass | 133 shards and exact manifest closure verified anonymously at revision `063c9e6f…` |
| Public baseline ARM64 image | pass | immutable index, ARM64 manifest, config, and all 54 layers fetched anonymously |
| Public DFlash ARM64 image | publication in progress | draft remains an external pinned CC-BY-NC-ND-4.0 download and is not bundled |
| Second-node clean load | pass | exact public snapshot and image, 89.89 GiB load, 1,638,400 KV tokens, CUDA graphs, health 200, behavior 5/5 |
| Abliteration | deferred | begins only after baseline publication is complete |

The release is intentionally not represented as meeting the decode-speed or
held-out-KLD gates. `verify_release.py` is strict: it returns nonzero until all
original requirements, including the speed target and publication pins, pass.
That behavior prevents a partial baseline from being mistaken for the original
full target.
