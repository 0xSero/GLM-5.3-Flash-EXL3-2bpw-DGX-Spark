import json
import unittest

import behavior_acceptance as acceptance


class BehaviorAcceptanceTests(unittest.TestCase):
    def test_exact_text_and_languages_require_natural_stop(self) -> None:
        expected = {"text": acceptance.TEXT_EXPECTED} | acceptance.LANGUAGE_CASES
        for name, content in expected.items():
            with self.subTest(name=name):
                record = {"name": name, "finish_reason": "stop", "content": content}
                self.assertTrue(acceptance.evaluate(record))
                self.assertFalse(
                    acceptance.evaluate(record | {"finish_reason": "length"})
                )
                self.assertFalse(acceptance.evaluate(record | {"content": content + "."}))

    def test_tool_call_requires_exact_name_and_arguments(self) -> None:
        record = {
            "name": "tools",
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"city": "Paris", "country": "France"}
                        ),
                    }
                }
            ],
        }
        self.assertTrue(acceptance.evaluate(record))
        self.assertFalse(acceptance.evaluate(record | {"finish_reason": "length"}))
        self.assertFalse(
            acceptance.evaluate(
                record
                | {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": "not-json",
                            }
                        }
                    ]
                }
            )
        )

    def test_request_contract_has_five_cases_and_no_token_cap(self) -> None:
        rows = acceptance.cases("served-model")
        self.assertEqual(
            [name for name, _ in rows],
            ["text", "tools", "arabic", "chinese", "polish"],
        )
        for _, payload in rows:
            self.assertEqual(payload["model"], "served-model")
            self.assertNotIn("max_tokens", payload)


if __name__ == "__main__":
    unittest.main()
