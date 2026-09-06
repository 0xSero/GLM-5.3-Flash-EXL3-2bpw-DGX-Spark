#!/usr/bin/env python3
"""Run short, natural-stop text, tool, and multilingual release checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


TEXT_EXPECTED = "FINAL_BASELINE_OK"
LANGUAGE_CASES = {
    "arabic": "مرحبا بالعالم",
    "chinese": "你好，世界",
    "polish": "Witaj, świecie",
}


def request(endpoint: str, payload: dict, timeout: float) -> tuple[dict, float]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    started = time.monotonic()
    with urllib.request.urlopen(
        urllib.request.Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        ),
        timeout=timeout,
    ) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def result_record(name: str, payload: dict, response: dict, seconds: float) -> dict:
    choice = response["choices"][0]
    message = choice["message"]
    record = {
        "name": name,
        "request_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "seconds": seconds,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "usage": response.get("usage"),
    }
    if name == "tools":
        record["tool_calls"] = message.get("tool_calls") or []
    return record


def evaluate(record: dict) -> bool:
    name = record["name"]
    if name != "tools" and record.get("finish_reason") != "stop":
        return False
    if name == "text":
        return record.get("content") == TEXT_EXPECTED
    if name in LANGUAGE_CASES:
        return record.get("content") == LANGUAGE_CASES[name]
    if name == "tools":
        if record.get("finish_reason") not in ("stop", "tool_calls"):
            return False
        calls = record.get("tool_calls") or []
        if len(calls) != 1:
            return False
        function = calls[0].get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "")
        except json.JSONDecodeError:
            return False
        return function.get("name") == "get_weather" and arguments == {
            "city": "Paris",
            "country": "France",
        }
    return False


def cases(model: str) -> list[tuple[str, dict]]:
    common = {"model": model, "temperature": 0.0}
    rows: list[tuple[str, dict]] = [
        (
            "text",
            common
            | {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly FINAL_BASELINE_OK as your final answer, "
                            "with no punctuation or surrounding text."
                        ),
                    }
                ]
            },
        ),
        (
            "tools",
            common
            | {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the weather tool for Paris, France.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city and country.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                    "country": {"type": "string"},
                                },
                                "required": ["city", "country"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        ),
    ]
    prompts = {
        "arabic": "أجب فقط بالعبارة التالية دون علامات اقتباس: مرحبا بالعالم",
        "chinese": "只回答以下文字，不要加引号或其他内容：你好，世界",
        "polish": "Odpowiedz wyłącznie tym tekstem, bez cudzysłowu: Witaj, świecie",
    }
    rows.extend(
        (name, common | {"messages": [{"role": "user", "content": prompt}]})
        for name, prompt in prompts.items()
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="glm-5.3-flash-exl3-k2-single-spark")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for name, payload in cases(args.model):
        response, seconds = request(args.endpoint, payload, args.timeout)
        record = result_record(name, payload, response, seconds)
        record["passed"] = evaluate(record)
        records.append(record)

    result = {
        "schema": "glm53-exact2-behavior-acceptance-v1",
        "model": args.model,
        "scope": "Short synthetic natural-stop release checks; not broad quality evaluation.",
        "results": records,
        "all_pass": all(record["passed"] for record in records),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_pass": result["all_pass"], "cases": len(records)}))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
