#!/usr/bin/env python3
"""Negative and positive controls for ``loop_acceptance.py``.

A repetition harness is only meaningful if it can fail. These controls build
synthetic cells with known defects and assert the harness rejects each one, and
that a correct cell still passes. No GPU or model is required.

Run::

    python3 tests/test_loop_acceptance.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("loop_acceptance", ROOT / "loop_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_tokens(text: str) -> list[int]:
    """Map text to plausible subword IDs.

    Real tokenizers emit subword units, so distinct-token ratios sit in the
    tenths, not the hundredths that one-character-per-token would produce. A
    stable mapping over fixed-width chunks reproduces that property without
    shipping a tokenizer.
    """
    chunks = [text[i : i + 2] for i in range(0, len(text), 2)]
    vocab: dict[str, int] = {}
    tokens: list[int] = []
    for chunk in chunks:
        if chunk not in vocab:
            vocab[chunk] = 1000 + len(vocab)
        tokens.append(vocab[chunk])
    return tokens


def make_cell(
    numbers: list[int] | None,
    *,
    text: str | None = None,
    token_ids: list[int] | None = None,
    finish_reason: str = "stop",
) -> dict:
    """Build a synthetic sweep cell in the recorded structured format."""
    if text is None:
        text = json.dumps({"numbers": numbers if numbers is not None else []})
    if token_ids is None:
        token_ids = synthetic_tokens(text)
    return {
        "target": 1024,
        "repeat": "control",
        "row": {
            "content": text,
            "finish_reason": finish_reason,
            "usage": {"completion_tokens": len(token_ids)},
            "events": [{"token_ids": token_ids, "content": text, "reasoning": ""}],
        },
    }


def main() -> int:
    module = _load_module()
    n = 300
    correct = list(range(1, n + 1))
    failures: list[str] = []

    def expect(name: str, cell: dict, want_pass: bool, want_check: str | None = None) -> None:
        """Assert a cell's verdict, and that the intended check drove it.

        For a negative control the named check must be False; for the positive
        control every check must be True. Asserting the *driver* keeps a cell
        from passing for an unrelated reason.
        """
        result = module.evaluate(cell, n)
        ok = result["passed"] is want_pass
        detail = ""
        if want_check is not None:
            got = result["checks"].get(want_check)
            ok = ok and got is (want_check is not None and (not want_pass or True) and got) and (got is False if not want_pass else True)
            detail = f" [{want_check}={got}]"
        verdict = "ok " if ok else "BAD"
        print(
            f"  {verdict} {name}: passed={result['passed']} "
            f"lrs={result['longest_repeated_token_ngram']} "
            f"run={result['longest_identical_token_run']}{detail}"
        )
        if not ok:
            failures.append(name)

    print("negative controls (harness must reject):")

    # 1. Classic degenerate loop: the model repeats a short cycle forever.
    cycle = [11, 22, 33, 44]
    loop_tokens = (cycle * 400)[:1600]
    expect(
        "infinite short-cycle loop",
        make_cell(None, text="1,2,3,4,5,6" * 200, token_ids=loop_tokens),
        want_pass=False,
        want_check="longest_repeated_ngram_within_limit",
    )

    # 2. Repetition collapse: one token repeated without bound.
    expect(
        "single-token repetition collapse",
        make_cell(None, text="7 " * 800, token_ids=[7] * 800),
        want_pass=False,
        want_check="identical_token_run_within_limit",
    )

    # 3. Text repeats a long verbatim block (a "looping" answer that is still
    #    well-formed JSON with the right shape).
    block = list(range(1, 101))
    expect(
        "verbatim block repeated in the value list",
        make_cell(block + block + block),
        want_pass=False,
        want_check="sequence_is_exact_1_to_n",
    )

    # 4. Duplicated values: the sequence stalls rather than progresses.
    stalled = correct[:150] + correct[:150]
    expect(
        "stalled / duplicated values",
        make_cell(stalled),
        want_pass=False,
        want_check="no_duplicate_values",
    )

    # 5. Truncated by a length cap instead of a natural stop.
    expect(
        "length-capped truncation",
        make_cell(correct[:120], finish_reason="length"),
        want_pass=False,
        want_check="natural_stop",
    )

    # 6. Near-constant output with a plausible finish reason.
    expect(
        "near-constant filler text",
        make_cell(None, text="a" * 3000, token_ids=[97] * 3000),
        want_pass=False,
        want_check="identical_token_run_within_limit",
    )

    print("positive control (harness must accept):")
    expect("correct 1..300 answer", make_cell(correct), want_pass=True)

    # Confirm the harness fails loudly when the control cells are absent.
    print("\nharness exit-code control:")
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.json"
        bad = Path(tmp) / "bad.json"
        good.write_text(json.dumps(make_cell(correct)))
        bad.write_text(json.dumps(make_cell(None, text="9 " * 900, token_ids=[9] * 900)))
        out = Path(tmp) / "out.json"
        rc_good = subprocess.run(
            [sys.executable, str(ROOT / "loop_acceptance.py"), str(good), "--output", str(out)],
            capture_output=True, text=True, check=False,
        ).returncode
        rc_bad = subprocess.run(
            [sys.executable, str(ROOT / "loop_acceptance.py"), str(bad), "--output", str(out)],
            capture_output=True, text=True, check=False,
        ).returncode
    print(f"  {'ok ' if rc_good == 0 else 'BAD'} correct cell exits 0 (got {rc_good})")
    print(f"  {'ok ' if rc_bad == 1 else 'BAD'} looping cell exits 1 (got {rc_bad})")
    if rc_good != 0:
        failures.append("exit-code good")
    if rc_bad != 1:
        failures.append("exit-code bad")

    print()
    if failures:
        print(f"FAILED: {len(failures)} control(s): {', '.join(failures)}")
        return 1
    print("All controls behaved as required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

def test_loop_acceptance_controls_pass() -> None:
    """Pytest entry point so the controls run with the rest of the suite."""
    assert main() == 0
