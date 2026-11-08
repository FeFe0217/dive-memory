"""Offline checks for the first DIVE pipeline; these are not Oracle evaluations."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dive_minimal import (  # noqa: E402
    ModelError,
    Store,
    answer_question,
    lexical_terms,
    make_units,
    select_units,
)


class FakeModel:
    """Capture outgoing prompts without reading credentials or using a network."""

    def __init__(self, response="根据证据，当前居住地是杭州。", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, messages, *, max_tokens, json_mode=False):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response, {
            "model": "offline-fake",
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
            "elapsed_seconds": 0.0,
        }


def sample_history():
    return {
        "sessions": [
            {
                "session_id": "residence-old",
                "date": "2026-09-10T09:00:00+08:00",
                "messages": [
                    {"role": "user", "content": "居住地：成都。"},
                    {"role": "assistant", "content": "知道了，你说居住地在成都。"},
                ],
            },
            {
                "session_id": "residence-new",
                "date": "2026-10-17T09:00:00+08:00",
                "messages": [
                    {"role": "user", "content": "居住地更新：10月1日搬到了杭州。"},
                    {"role": "assistant", "content": "你说当前居住地是杭州。"},
                ],
            },
            {
                "session_id": "unrelated",
                "date": "2026-10-18T09:00:00+08:00",
                "messages": [
                    {
                        "role": "user",
                        "content": "NEVER_SEND_PRIVATE_NOTE xylophone，上海出差。",
                    }
                ],
            },
        ]
    }


class MakeUnitsTests(unittest.TestCase):
    def test_orders_dates_and_preserves_each_source_role(self):
        earlier_messages = [
            {"role": "assistant", "content": "这是一条独立的助手消息。"},
            {"role": "user", "content": "第一轮用户信息。"},
            {"role": "assistant", "content": "第一次助手回复。"},
            {"role": "assistant", "content": "接着补充一句。"},
            {"role": "user", "content": "第二轮用户信息。"},
        ]
        later_messages = [{"role": "user", "content": "较晚的用户信息。"}]
        units = make_units(
            {
                "sessions": [
                    {
                        "session_id": "late",
                        "date": "2026-10-02T08:00:00+08:00",
                        "messages": later_messages,
                    },
                    {
                        "session_id": "early",
                        "date": "2026-10-01T08:00:00+08:00",
                        "messages": earlier_messages,
                    },
                ]
            }
        )
        self.assertEqual(units[0]["session_id"], "early")
        self.assertEqual(units[-1]["session_id"], "late")
        self.assertEqual(len({unit["unit_id"] for unit in units}), len(units))
        flattened = [message for unit in units for message in unit["messages"]]
        self.assertEqual(flattened, earlier_messages + later_messages)
        self.assertEqual(len([u for u in units if u["session_id"] == "early"]), 3)

    def test_rejects_empty_content_duplicate_session_and_invalid_date(self):
        for bad_content in ("", " \t"):
            history = sample_history()
            history["sessions"][0]["messages"][0]["content"] = bad_content
            with self.subTest(content=repr(bad_content)), self.assertRaises(
                (ValueError, TypeError)
            ):
                make_units(history)
        history = sample_history()
        history["sessions"].append(copy.deepcopy(history["sessions"][0]))
        with self.subTest(case="duplicate session"), self.assertRaises(
            (ValueError, TypeError)
        ):
            make_units(history)
        history = sample_history()
        history["sessions"][0]["date"] = "not-an-ISO-date"
        with self.subTest(case="invalid date"), self.assertRaises((ValueError, TypeError)):
            make_units(history)


class SelectionTests(unittest.TestCase):
    def test_selection_keeps_original_source_not_generated_rewrite(self):
        units = make_units(sample_history())
        original = copy.deepcopy(units)
        selected_id = units[1]["unit_id"]
        model = FakeModel(json.dumps({"keep_ids": [selected_id]}))
        selected, call_info = select_units(units, model)
        self.assertEqual(selected, [original[1]])
        self.assertEqual(units, original)
        self.assertIsInstance(call_info, dict)
        self.assertEqual(len(model.calls), 1)

    def test_selection_rejects_invalid_json_types_duplicate_and_hallucinated_ids(self):
        units = make_units(sample_history())
        unit_id = units[0]["unit_id"]
        invalid_responses = [
            "not JSON",
            "[]",
            json.dumps({"other": [unit_id]}),
            json.dumps({"keep_ids": unit_id}),
            json.dumps({"keep_ids": [123]}),
            json.dumps({"keep_ids": [unit_id, unit_id]}),
            json.dumps({"keep_ids": [unit_id, "invented-unit-id"]}),
        ]
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ModelError):
                select_units(units, FakeModel(response))


class StoreTests(unittest.TestCase):
    def test_database_survives_reopen_and_namespaces_are_isolated(self):
        units = make_units(sample_history())
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            database = Path(directory) / "memory.sqlite3"
            with Store(database) as store:
                self.assertEqual(store.write("alice", units), len(units))
                store.write("bob", [units[2]])
            with Store(database) as reopened:
                self.assertEqual(reopened.count("alice"), len(units))
                self.assertEqual(reopened.count("bob"), 1)
                self.assertEqual(reopened.count("missing"), 0)
                self.assertTrue(reopened.search("alice", "居住地"))
                self.assertEqual(reopened.search("bob", "居住地"), [])

    def test_chinese_terms_include_overlapping_bigrams(self):
        terms = lexical_terms("我的居住地是杭州")
        self.assertIn("居住", terms)
        self.assertIn("住地", terms)
        self.assertIn("杭州", terms)

    def test_ascii_terms_are_lowercase_words_and_numbers(self):
        terms = lexical_terms("ALPHA Beta42 123")
        self.assertIn("alpha", terms)
        self.assertIn("beta42", terms)
        self.assertIn("123", terms)

    def test_chinese_matching_and_zero_hits_without_history_fallback(self):
        units = make_units(sample_history())
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            with Store(Path(directory) / "memory.sqlite3") as store:
                store.write("demo", units)
                evidence = store.search("demo", "现在的居住地在哪里")
                self.assertTrue(evidence)
                self.assertTrue(
                    all(unit["session_id"].startswith("residence-") for unit in evidence)
                )
                self.assertEqual(store.search("demo", "火星基地库存"), [])

    def test_search_respects_result_limit_and_original_text_budget(self):
        units = make_units(sample_history())
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            with Store(Path(directory) / "memory.sqlite3") as store:
                store.write("demo", units)
                self.assertEqual(len(store.search("demo", "居住地", limit=1)), 1)
                matching = [unit for unit in units if unit["session_id"].startswith("residence-")]
                budget = max(len(json.dumps(unit, ensure_ascii=False)) for unit in matching)
                evidence = store.search("demo", "居住地", limit=3, max_chars=budget)
                self.assertTrue(evidence)
                total_chars = sum(
                    len(json.dumps(unit, ensure_ascii=False)) for unit in evidence
                )
                self.assertLessEqual(total_chars, budget)
                originals = {unit["unit_id"]: unit for unit in units}
                for unit in evidence:
                    self.assertEqual(unit["messages"], originals[unit["unit_id"]]["messages"])


class AnswerTests(unittest.TestCase):
    def test_answer_prompt_contains_retrieved_evidence_not_unrelated_history(self):
        units = make_units(sample_history())
        model = FakeModel()
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            with Store(Path(directory) / "memory.sqlite3") as store:
                store.write("demo", units)
                answer, evidence, call_info = answer_question(
                    store, "demo", "现在的居住地在哪里？", "2026-10-24T13:00:00+08:00", model
                )
        self.assertEqual(answer, model.response)
        self.assertTrue(evidence)
        self.assertIsInstance(call_info, dict)
        self.assertEqual(len(model.calls), 1)
        prompt = json.dumps(model.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("现在的居住地在哪里？", prompt)
        for unit in evidence:
            for message in unit["messages"]:
                self.assertIn(message["content"], prompt)
        self.assertNotIn("NEVER_SEND_PRIVATE_NOTE", prompt)

    def test_zero_evidence_abstains_without_a_model_request(self):
        model = FakeModel(error=AssertionError("A zero-hit query must not call the model"))
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            with Store(Path(directory) / "memory.sqlite3") as store:
                store.write("demo", make_units(sample_history()))
                answer, evidence, call_info = answer_question(
                    store, "demo", "火星基地库存是多少？", "2026-10-24T13:00:00+08:00", model
                )
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())
        self.assertEqual(evidence, [])
        self.assertIsInstance(call_info, dict)
        self.assertEqual(model.calls, [])

    def test_model_failure_is_not_converted_to_normal_abstention(self):
        model = FakeModel(error=RuntimeError("simulated upstream failure"))
        with tempfile.TemporaryDirectory(prefix="dive-offline-test-") as directory:
            with Store(Path(directory) / "memory.sqlite3") as store:
                store.write("demo", make_units(sample_history()))
                with self.assertRaisesRegex(RuntimeError, "simulated upstream failure"):
                    answer_question(
                        store, "demo", "现在的居住地在哪里？", "2026-10-24T13:00:00+08:00", model
                    )


if __name__ == "__main__":
    unittest.main()
