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
| Text natural stop | pass | exact response on final baseline |
| Structured tool call | pass | function name and JSON arguments parsed |
| Multilingual smoke | pass | Arabic, Chinese, Polish |
| Images | pass | 4/4 exact controlled fixtures in `evidence/vision-final.json` |
| Native video | pass | 2/2 exact controlled fixtures in `evidence/vision-final.json` |
| Real 200k request | pass | 200,012 server prompt tokens and 4/4 exact retrieval in `evidence/long-context-final.json` |
| 25–50 tok/s target | **fail** | bounded estimates cluster near 9.3 tok/s |
| Code responsiveness | **fail** | simple Python request had no final content by 600 seconds |
| Exact held-out KLD | unmeasured | BF16 comparison was not feasible in the remaining one-Spark publication window |
| DFlash2 | rejected | initialization failed; not bundled or advertised |
| Public HF weights | in progress | repository exists; immutable anonymous verification follows upload |
| Public ARM64 image | in progress | registry upload and anonymous verification follow |
| Second-node clean pull/load | pending | requires another idle Spark and does not replace the already accepted live baseline |
| Abliteration | deferred | begins only after baseline publication is complete |

The release is intentionally not represented as meeting the decode-speed or
held-out-KLD gates. `verify_release.py` is strict: it returns nonzero until all
original requirements, including the speed target and publication pins, pass.
That behavior prevents a partial baseline from being mistaken for the original
full target.

