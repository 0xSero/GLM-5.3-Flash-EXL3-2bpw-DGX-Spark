#!/usr/bin/env python3
"""Degenerate-repetition (looping) acceptance for natural-stop generations.

A looping or repetition-collapsed decoder cannot satisfy a task that requires a
long run of distinct, ordered values. This harness therefore combines a
task-structure check with model-independent repetition statistics computed over
the exact emitted token IDs.

Checks per cell:

  structured    the response is a single JSON object whose ``numbers`` array is
                exactly ``1..N`` once each, in order
  termination   the stream ended with a natural ``stop``, not a length cap
  repetition    longest repeated token n-gram, longest run of one repeated
                token, distinct-token ratio, and gzip ratio of the text

Every threshold is reported numerically; the pass/fail verdicts are deliberately
conservative so a partial run cannot be mistaken for an accepted one.

Reads the ``structured`` sweep cell format: each cell is a JSON document with a
top-level ``row`` holding ``content``, ``events`` (each with ``token_ids``) and
``finish_reason``. See ``BENCHMARKS-NATIVE-MTP-262K.md`` for the producing run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

# A true 300-value ascending sequence has no long repeated token n-gram. These
# limits sit far above what a correct response measures and far below a loop.
MAX_REPEATED_NGRAM_TOKENS = 64
MAX_IDENTICAL_TOKEN_RUN = 24
MIN_DISTINCT_TOKEN_RATIO = 0.05

# Repetition collapse makes output *more* compressible, so gzip is bounded from
# below, not above. It is a secondary signal here: a correct answer to this
# counting task is itself highly patterned (300 comma-separated integers), so
# healthy output measures a middling ratio and only a near-constant stream
# falls through the floor.
MIN_GZIP_RATIO = 0.15

NUMBER_RE = re.compile(r"-?\d+")


def _token_ids(cell: dict) -> list[int]:
    ids: list[int] = []
    for event in cell.get("row", {}).get("events", []):
        for token in event.get("token_ids") or []:
            if isinstance(token, int):
                ids.append(token)
    return ids


def _reasoning_ids(cell: dict) -> list[int]:
    ids: list[int] = []
    for event in cell.get("row", {}).get("events", []):
        if (event.get("reasoning") or "").strip() and event.get("token_ids"):
            for token in event["token_ids"]:
                if isinstance(token, int):
                    ids.append(token)
    return ids


def longest_repeated_ngram(tokens: list[int], cap: int) -> int:
    """Length (in tokens) of the longest n-gram appearing at least twice."""
    n = len(tokens)
    if n < 2:
        return 0
    low, high, best = 1, min(cap, n - 1), 0
    while low <= high:
        mid = (low + high) // 2
        seen: set[tuple[int, ...]] = set()
        found = False
        for i in range(n - mid + 1):
            gram = tuple(tokens[i : i + mid])
            if gram in seen:
                found = True
                break
            seen.add(gram)
        if found:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def longest_identical_run(tokens: list[int]) -> int:
    best = run = 0
    previous = None
    for token in tokens:
        if token == previous:
            run += 1
        else:
            run = 1
            previous = token
        best = max(best, run)
    return best


def evaluate(cell: dict, expected_count: int) -> dict:
    row = cell.get("row", {})
    content = row.get("content") or ""
    finish_reason = row.get("finish_reason")
    ids = _token_ids(cell)
    reasoning = _reasoning_ids(cell)

    numbers: list[int] | None = None
    parse_error: str | None = None
    try:
        parsed = json.loads(content)
        candidate = parsed.get("numbers") if isinstance(parsed, dict) else None
        if isinstance(candidate, list) and all(isinstance(v, int) for v in candidate):
            numbers = candidate
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        parse_error = f"{type(exc).__name__}: {exc}"

    expected = list(range(1, expected_count + 1))
    sequence_exact = numbers == expected

    text_tokens = NUMBER_RE.findall(content)
    gzip_ratio = None
    if content:
        gzip_ratio = round(len(gzip.compress(content.encode())) / len(content.encode()), 6)

    distinct_ratio = round(len(set(ids)) / len(ids), 6) if ids else 0.0
    ngram = longest_repeated_ngram(ids, MAX_REPEATED_NGRAM_TOKENS) if ids else 0
    ngram_cap_hit = bool(ids) and ngram >= MAX_REPEATED_NGRAM_TOKENS
    run = longest_identical_run(ids) if ids else 0

    checks = {
        "json_object_parsed": numbers is not None,
        "sequence_is_exact_1_to_n": bool(sequence_exact),
        "no_duplicate_values": bool(numbers is not None and len(set(numbers)) == len(numbers)),
        "strictly_increasing": bool(
            numbers is not None and all(b > a for a, b in zip(numbers, numbers[1:]))
        ),
        "natural_stop": finish_reason == "stop",
        "longest_repeated_ngram_within_limit": ngram < MAX_REPEATED_NGRAM_TOKENS,
        "identical_token_run_within_limit": run <= MAX_IDENTICAL_TOKEN_RUN,
        "distinct_token_ratio_above_floor": distinct_ratio >= MIN_DISTINCT_TOKEN_RATIO,
        "gzip_ratio_above_floor": gzip_ratio is None or gzip_ratio >= MIN_GZIP_RATIO,
    }

    return {
        "cell": None,  # filled by caller
        "target_input_tokens": cell.get("target") or cell.get("actual_prompt_tokens"),
        "repeat": cell.get("repeat"),
        "finish_reason": finish_reason,
        "completion_tokens": row.get("usage", {}).get("completion_tokens"),
        "emitted_token_ids": len(ids),
        "reasoning_token_ids": len(reasoning),
        "numbers_emitted": len(numbers) if numbers else 0,
        "numeric_literals_in_text": len(text_tokens),
        "longest_repeated_token_ngram": ngram,
        "repeated_ngram_cap_hit": ngram_cap_hit,
        "longest_identical_token_run": run,
        "distinct_token_ratio": distinct_ratio,
        "gzip_ratio": gzip_ratio,
        "parse_error": parse_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", metavar="CELL.json")
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for path_str in args.cells:
        path = Path(path_str)
        cell = json.loads(path.read_text())
        result = evaluate(cell, args.expected_count)
        result["cell"] = path.name
        result["cell_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(result)

    summary = {
        "schema": "glm53-loop-acceptance-v1",
        "expected_sequence": f"1..{args.expected_count}",
        "thresholds": {
            "max_repeated_ngram_tokens": MAX_REPEATED_NGRAM_TOKENS,
            "max_identical_token_run": MAX_IDENTICAL_TOKEN_RUN,
            "min_distinct_token_ratio": MIN_DISTINCT_TOKEN_RATIO,
            "min_gzip_ratio": MIN_GZIP_RATIO,
        },
        "scope": (
            "Degenerate-repetition acceptance over exact emitted token IDs for the "
            "structured counting task. It detects looping and repetition collapse. "
            "It is not a broad quality, correctness, or reasoning evaluation, and a "
            "pass here does not imply any other gate passed."
        ),
        "cells": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "all_pass": bool(results) and all(r["passed"] for r in results),
        "results": results,
    }

    def _clean(value):
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items() if k != "cell"}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    Path(args.output).write_text(json.dumps(_clean(summary), indent=2) + "\n")
    print(
        f"loop acceptance: {summary['passed']}/{summary['cells']} passed, "
        f"all_pass={summary['all_pass']} -> {args.output}"
    )
    for result in results:
        worst_ngram = result["longest_repeated_token_ngram"]
        print(
            f"  {str(result['cell']):>26}  in={str(result['target_input_tokens']):>7}  "
            f"tok={result['emitted_token_ids']:>5}  lrs_ngram={worst_ngram:>3}  "
            f"run={result['longest_identical_token_run']:>3}  "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())