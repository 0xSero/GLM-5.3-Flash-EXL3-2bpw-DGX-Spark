# Benchmark evidence

All published measurements are finite natural-stop requests. No request output
token cap was sent. Client decode rate is computed from final usage counts and
the interval from first to last streamed content; it is not a kernel metric.

| Input text tokens | Incoming concurrency | Wall time | Decode estimate | Result |
|---:|---:|---:|---:|---|
| 1,024 | 1 | 14.88 s | 9.16 tok/s | pass |
| 32,768 | 1 | 84.45 s | 9.47 tok/s | pass |
| 131,072 | 1 | 306.10 s | 9.28 tok/s | pass |
| 1,024 | 2 | 29.84 s total | 9.34 tok/s mean | 2/2 pass, serialized |

The server intentionally used `max_num_seqs=1`; concurrency two represents two
incoming clients, not simultaneous decoding. Raw results are in
`evidence/bounded-speed-c1.json` and `evidence/bounded-speed-c2.json`.

The long-context acceptance request used fresh randomized records and prefix
caching was disabled. It contained 200,000 input text tokens and 200,012
server-reported prompt tokens, retrieved four of four values exactly, stopped
naturally, and took 498.40 seconds. The raw result is
`evidence/long-context-final.json`.

The original broader matrix and five-repeat 25–50 tok/s target were not
completed after the bounded cells established that this baseline is around
9.3 tok/s. More repetition would spend GPU credits without changing that
release decision. Unsupported or unrun cells are not presented as passes.

