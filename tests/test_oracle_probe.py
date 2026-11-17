"""Offline protocol checks: synthetic fixtures only, no model downloads or fees."""

import copy
from contextlib import closing, redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dive_minimal import ModelError  # noqa: E402
import oracle_probe  # noqa: E402
from oracle_probe import (  # noqa: E402
    TrackedModel,
    answer_from_evidence,
    format_embedding_text,
    make_batches,
    make_probe_units,
    parse_oracle_date,
    rank_evidence,
    sanitize_history,
)


class FakeModel:
    def __init__(self, response=None, error=None):
        self.response = response or json.dumps({
            "answer": "Hangzhou.", "used_ids": ["u0000"], "status": "answered",
        })
        self.error = error
        self.calls = []
        self._key = "NEVER_LOG_A_CREDENTIAL"

    def complete(self, messages, *, max_tokens, json_mode=False):
        self.calls.append(copy.deepcopy(messages))
        if self.error:
            raise self.error
        return self.response, {
            "provider": "offline-fake", "model": "offline-fake", "seconds": 0.0,
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        }


def fixture():
    return {
        "question_id": "example_abs", "question_type": "TYPE_LABEL_CANARY",
        "question": "FUTURE_QUESTION_CANARY", "answer": "REFERENCE_ANSWER_CANARY",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": ["answer_late_abs", "answer_early"],
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/10 (Mon) 14:47"],
        "haystack_sessions": [
            [{"role": "user", "content": "Later source.", "has_answer": "LOCATION_LABEL_CANARY"}],
            [{"role": "assistant", "content": "Earlier source.", "has_answer": True}],
        ],
        "answer_session_ids": ["answer_late_abs"],
    }


def unit(uid="u0000", content="I live in Hangzhou."):
    return {
        "unit_id": uid, "session_id": "s0", "date": "2023-04-10T14:47:00",
        "messages": [{"role": "user", "content": content,
                      "message_index": 0, "start": 0, "end": len(content)}],
    }


def values_for_key(value, key):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                yield child
            yield from values_for_key(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from values_for_key(child, key)


class OracleProbeTests(unittest.TestCase):
    def test_date_sort_keeps_sources_aligned_and_excludes_labels(self):
        raw = fixture()
        history, mapping = sanitize_history(raw)
        sessions = history["sessions"]
        self.assertEqual([s["messages"][0]["content"] for s in sessions],
                         ["Earlier source.", "Later source."])
        for session in sessions:
            self.assertIsNone(datetime.fromisoformat(session["date"]).tzinfo)
            for message in session["messages"]:
                self.assertEqual(set(message), {"role", "content"})
        outgoing = json.dumps(history)
        for forbidden in ("FUTURE_QUESTION_CANARY", "REFERENCE_ANSWER_CANARY",
                          "TYPE_LABEL_CANARY", "LOCATION_LABEL_CANARY", "answer_late_abs", "answer_early"):
            self.assertNotIn(forbidden, outgoing)
        self.assertIn("answer_late_abs", json.dumps(mapping))
        parsed = parse_oracle_date("2023/04/10 (Mon) 14:47")
        self.assertEqual(datetime.fromisoformat(parsed), datetime(2023, 4, 10, 14, 47))
        with self.assertRaises(ValueError):
            parse_oracle_date("2023/02/30 (Mon) 14:47")

    def test_long_messages_can_be_reconstructed_exactly_from_spans(self):
        messages = [
            {"role": "assistant", "content": "An opening assistant message."},
            {"role": "user", "content": "Long text with spaces. 中文字符。" * 12},
            {"role": "assistant", "content": "Follow-up details. " * 10},
            {"role": "user", "content": "Another user turn."},
        ]
        history = {"sessions": [{"session_id": "s0", "date": "2023-04-10T14:47:00", "messages": messages}]}
        units = make_probe_units(history, len, max_tokens=64)
        self.assertGreater(len(units), 4)
        self.assertEqual(len({u["unit_id"] for u in units}), len(units))
        reconstructed = {i: [] for i in range(len(messages))}
        for part in units:
            self.assertLessEqual(len(format_embedding_text(part)), 64)
            for message in part["messages"]:
                original = messages[message["message_index"]]
                self.assertEqual(message["role"], original["role"])
                self.assertEqual(message["content"], original["content"][message["start"]:message["end"]])
                reconstructed[message["message_index"]].append(message["content"])
        self.assertEqual(["".join(reconstructed[i]) for i in range(len(messages))],
                         [m["content"] for m in messages])
        annotated = unit()
        annotated["source_label"] = "EXTERNAL_METADATA_CANARY"
        self.assertNotIn("EXTERNAL_METADATA_CANARY", format_embedding_text(annotated))

    def test_batches_obey_count_and_serialized_length_without_dropping_units(self):
        units = [unit(f"u{i:04d}", "Content " * (i + 1)) for i in range(5)]
        ceiling = max(len(json.dumps([u], ensure_ascii=False)) for u in units) + 50
        batches = make_batches(units, max_units=2, max_chars=ceiling)
        self.assertEqual([u for batch in batches for u in batch], units)
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False)), ceiling)
        with self.assertRaises(ValueError):
            make_batches([units[0]], max_chars=len(json.dumps([units[0]], ensure_ascii=False)) - 1)

    def test_semantic_ranking_returns_original_evidence_within_budget(self):
        units = [unit("u0000", "Unrelated."), unit("u0001", "Relevant."), unit("u0002", "Somewhat related.")]
        vectors = [[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]]
        evidence, _ = rank_evidence(units, vectors, [1.0, 0.0], top_k=2, max_chars=10000)
        self.assertEqual([u["unit_id"] for u in evidence], ["u0001", "u0002"])
        budget = len(json.dumps(units[1], ensure_ascii=False))
        evidence, _ = rank_evidence(units, vectors, [1.0, 0.0], top_k=3, max_chars=budget)
        self.assertEqual(evidence, [units[1]])

    def test_answer_uses_question_and_evidence_and_returns_plain_hypothesis(self):
        model = FakeModel()
        answer, details = answer_from_evidence("Where do I live?", "2023-04-10T23:07:00", [unit()], model)
        self.assertEqual(answer, "Hangzhou.")
        self.assertIsInstance(details, dict)
        prompt = json.dumps(model.calls[0], ensure_ascii=False)
        self.assertIn("Where do I live?", prompt)
        self.assertIn("I live in Hangzhou.", prompt)

    def test_answer_rejects_fabricated_duplicate_or_missing_citations(self):
        for payload in (
            {"answer": "Hangzhou.", "used_ids": ["invented"], "status": "answered"},
            {"answer": "Hangzhou.", "used_ids": ["u0000", "u0000"], "status": "answered"},
            {"answer": "Hangzhou.", "used_ids": [], "status": "answered"},
            {"answer": "Hangzhou.", "used_ids": ["u0000"], "status": "unknown-status"},
            {"answer": " ", "used_ids": ["u0000"], "status": "answered"},
        ):
            with self.subTest(payload=payload), self.assertRaises((ModelError, ValueError)):
                answer_from_evidence("Where?", "2023-04-10T23:07:00", [unit()], FakeModel(json.dumps(payload)))

    def test_empty_evidence_skips_model_but_provider_failure_propagates(self):
        model = FakeModel(error=RuntimeError("offline provider failure"))
        answer, _ = answer_from_evidence("Where?", "2023-04-10T23:07:00", [], model)
        self.assertTrue(answer.strip())
        self.assertEqual(model.calls, [])
        with self.assertRaisesRegex(RuntimeError, "offline provider failure"):
            answer_from_evidence("Where?", "2023-04-10T23:07:00", [unit()], model)

    def test_usage_survives_output_validation_failure_and_call_cap_is_enforced(self):
        with tempfile.TemporaryDirectory(prefix="dive-oracle-offline-") as folder:
            trace = Path(folder) / "calls.jsonl"
            base = FakeModel("not JSON")
            tracked = TrackedModel(base, trace, max_calls=1)
            tracked.stage = "answer"
            with self.assertRaises((ModelError, ValueError)):
                answer_from_evidence("Where?", "2023-04-10T23:07:00", [unit()], tracked)
            self.assertEqual(len(tracked.calls), 1)
            self.assertIn(24, list(values_for_key(tracked.calls, "total_tokens")))
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertIn(24, list(values_for_key(rows, "total_tokens")))
            with self.assertRaises((ModelError, ValueError, RuntimeError)):
                tracked.complete([], max_tokens=16)
            self.assertEqual(len(base.calls), 1)

    def test_failed_call_has_unknown_usage_and_never_logs_credentials(self):
        with tempfile.TemporaryDirectory(prefix="dive-oracle-offline-") as folder:
            trace = Path(folder) / "calls.jsonl"
            tracked = TrackedModel(FakeModel(error=RuntimeError("offline failure")), trace, max_calls=1)
            with self.assertRaises(RuntimeError):
                tracked.complete([], max_tokens=16)
            totals = list(values_for_key(tracked.calls, "total_tokens"))
            self.assertNotIn(0, totals)
            self.assertNotIn("NEVER_LOG_A_CREDENTIAL", trace.read_text(encoding="utf-8"))

    def test_main_persists_prediction_usage_and_evidence_with_only_fake_models(self):
        class Tokenizer:
            def encode(self, text):
                return list(text)

        class Array:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return copy.deepcopy(self.values)

            def __getitem__(self, index):
                return Array(self.values[index])

        class Encoder:
            def encode(self, texts, **kwargs):
                return Array([[1.0] + [0.0] * 1023 for _ in texts])

        class ProbeModel(FakeModel):
            def __init__(self, key, model_name):
                super().__init__()

            def complete(self, messages, *, max_tokens, json_mode=False):
                payload = json.loads(messages[-1]["content"])
                if isinstance(payload, list):
                    self.response = json.dumps({"keep_ids": [u["unit_id"] for u in payload]})
                else:
                    self.response = json.dumps({
                        "answer": "Hangzhou.", "used_ids": ["u0000"], "status": "answered",
                    })
                return super().complete(messages, max_tokens=max_tokens, json_mode=json_mode)

        with tempfile.TemporaryDirectory(prefix="dive-oracle-integration-") as folder:
            root = Path(folder)
            data = root / "synthetic.json"
            item = fixture()
            item["question"] = "Where?"
            data.write_text(json.dumps([item]), encoding="utf-8")
            with (patch.object(oracle_probe, "ROOT", root),
                  patch.object(oracle_probe, "cached_tokenizer", return_value=Tokenizer()),
                  patch.object(oracle_probe, "cached_encoder", return_value=Encoder()),
                  patch.object(oracle_probe, "read_key", return_value="offline-test-key"),
                  patch.object(oracle_probe, "DeepSeekModel", ProbeModel),
                  redirect_stdout(io.StringIO())):
                result = oracle_probe.main(["run", "--data", str(data), "--limit", "1", "--max-calls", "2"])
            self.assertEqual(result, 0)
            run_folder, = (root / "runs" / "oracle-probe").iterdir()
            summary = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
            predictions = [json.loads(line) for line in
                           (run_folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(predictions, [{"question_id": item["question_id"], "hypothesis": "Hangzhou."}])
            self.assertIsInstance(predictions[0]["hypothesis"], str)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(summary["calls"]), 2)
            self.assertEqual(summary["usage"]["known_usage_totals"]["total_tokens"], 48)
            with closing(sqlite3.connect(run_folder / "memory.sqlite3")) as database:
                self.assertGreater(database.execute("SELECT count(*) FROM evidence").fetchone()[0], 0)

            item["question"] = ""
            data.write_text(json.dumps([item]), encoding="utf-8")
            with (patch.object(oracle_probe, "ROOT", root),
                  patch.object(oracle_probe, "cached_tokenizer", return_value=Tokenizer()),
                  patch.object(oracle_probe, "cached_encoder") as encoder,
                  patch.object(oracle_probe, "read_key") as read_key,
                  patch.object(oracle_probe, "DeepSeekModel") as model,
                  redirect_stdout(io.StringIO())):
                result = oracle_probe.main(["run", "--data", str(data), "--limit", "1", "--max-calls", "2"])
            self.assertEqual(result, 1)
            read_key.assert_not_called()
            model.assert_not_called()
            encoder.assert_not_called()


ROOT = Path(__file__).resolve().parents[1]


def source_claim(cid, source, value="Chengdu", **extra):
    message = source["messages"][0]
    claim = {"claim_id": cid, "unit_id": source["unit_id"], "kind": "state", "subject": "user",
             "attribute": "residence", "value": value, "scope": "primary residence",
             "modality": "actual", "polarity": "positive", "temporal_mode": "current",
             "time_lower": None, "time_upper": None, "time_expression": None,
             "observed_at": source["date"], "speaker": message["role"],
             "source_session_id": source["session_id"], "source_message_index": message["message_index"],
             "refs": [{"unit_id": source["unit_id"], "message_index": message["message_index"],
                       "start": message["start"], "end": message["end"], "quote": message["content"], "purpose": "claim"}]}
    claim.update(extra)
    return claim


def proposal(claim, action="ADD", old_ids=None, **extra):
    result = {"claim_id": claim["claim_id"], "action": action, "old_ids": old_ids or [],
              "same_slot": None, "same_event": None, "value_relation": "unknown",
              "basis_refs": copy.deepcopy(claim["refs"]), "reason": "Direct source statement."}
    result.update(extra)
    return result


class OfflineEncoder:
    class Array:
        def __init__(self, rows): self.rows = rows
        def tolist(self): return self.rows

    def encode(self, texts, **kwargs):
        return self.Array([[1.0] + [0.0] * 1023 for _ in texts])


class UnifiedPipelineTests(unittest.TestCase):
    def tempdir(self, prefix):
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=parent)

    def test_answer_failure_is_captured_before_parsing_with_known_usage(self):
        with self.tempdir("reader-capture-") as folder:
            tracked = TrackedModel(FakeModel("BROKEN_ANSWER_JSON"), Path(folder) / "calls.jsonl", 1,
                                   capture_validation=True)
            tracked.stage = "oracle/answer"
            with redirect_stdout(io.StringIO()), self.assertRaises(ModelError):
                answer_from_evidence("Where?", None, [unit()], tracked)
            entries = [json.loads(line) for line in (Path(folder) / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(entries[0]["stage"], "ANSWER")
            self.assertEqual(entries[0]["raw_output"], "BROKEN_ANSWER_JSON")
            self.assertEqual(entries[0]["call_info"]["usage"]["total_tokens"], 24)
            self.assertFalse(entries[0]["replayable"])

    def test_usage_missing_derived_and_inconsistent_are_not_complete(self):
        from oracle_probe import usage_summary
        raw = {"prompt_tokens": 100, "completion_tokens": 20}
        before = copy.deepcopy(raw)
        result = usage_summary([{"usage": raw}])
        self.assertEqual(raw, before)
        self.assertEqual(result["known_usage_totals"]["total_tokens"], 120)
        self.assertIsNone(result["known_usage_totals"]["prompt_cache_hit_tokens"])
        self.assertFalse(result["usage_complete"])
        self.assertTrue(result["derived_fields"])
        result = usage_summary([{"usage": dict(raw, total_tokens=999,
                                               prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=80)}])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(len(result["inconsistencies"]), 2)

    def test_context_sources_expand_their_own_correction_dependencies(self):
        from oracle_probe import _design_answer_evidence
        sources = [unit("a", "Atlas question."), unit("b", "I lived in Nanjing."),
                   unit("c", "Nanjing was wrong; I lived in Hangzhou.")]
        claims = [source_claim("A", sources[0], "Atlas"), source_claim("B", sources[1], "Nanjing", validity="corrected"),
                  source_claim("C", sources[2], "Hangzhou")]
        claims[0]["refs"].append(dict(claims[1]["refs"][0], purpose="context"))
        records = [{"record_id": c["claim_id"], "support_ids": [c["claim_id"]]} for c in claims]
        relation = {"action": "CORRECT", "old_ids": ["B"], "new_id": "C", "same_slot": True,
                    "target_claim_ids": ["B"], "basis_refs": claims[1]["refs"] + claims[2]["refs"]}
        stored = {"retained_units": sources, "claims": claims, "records": records, "relations": [relation],
                  "vectors": [[1.0] + [0.0] * 1023] * 3}
        with patch.object(oracle_probe, "_top_raw_ids", return_value=[{"unit_id": "a", "score": 1.0}]):
            evidence, _ = _design_answer_evidence("Where?", stored, OfflineEncoder(), len)
        self.assertEqual({u["unit_id"] for u in evidence}, {"a", "b", "c"})
        self.assertTrue(all(not e["coverage_incomplete"] for e in evidence))

    def test_all_observations_used_by_time_guard_reach_reader(self):
        from dive_memory import apply_updates, judge_updates
        from oracle_probe import _design_answer_evidence
        sources = [unit("a", "I live in Chengdu."), unit("b", "I still live in Chengdu."),
                   unit("s", "I live in Hangzhou.")]
        for source, date in zip(sources, ("2026-09-01", "2026-09-10", "2026-09-05")):
            source["date"] = date
        a, b, s = [source_claim(cid, u, value) for cid, u, value in
                   zip(("A", "B", "S"), sources, ("Chengdu", "Chengdu", "Hangzhou"))]
        records = [{"record_id": "A", "support_ids": ["A"]}]
        support = proposal(b, "SUPPORT", ["A"], same_slot=True)
        merged = apply_updates(records, [b], [support], claims=[a], relations=[])
        change = proposal(s, "CHANGE", ["A"], same_slot=True, value_relation="exclusive")
        model = FakeModel(json.dumps({"operations": [change]}))
        judged = judge_updates([s], merged["records"], {"S": ["A"]}, model,
                               claims=merged["claims"], relations=merged["relations"])
        self.assertEqual(judged["proposals"][0]["action"], "UNCERTAIN")
        result = apply_updates(merged["records"], [s], judged["proposals"],
                               claims=merged["claims"], relations=merged["relations"])
        stored = dict(result, retained_units=sources, relations=merged["relations"] + result["relations"],
                      vectors=[[1.0] + [0.0] * 1023] * 3)
        with patch.object(oracle_probe, "_top_raw_ids", return_value=[{"unit_id": "s", "score": 1.0}]):
            evidence, _ = _design_answer_evidence("Where now?", stored, OfflineEncoder(), len)
        self.assertEqual({u["unit_id"] for u in evidence}, {"a", "b", "s"})

    def test_evidence_budget_uses_complete_answer_request_and_whole_groups(self):
        from oracle_probe import _design_answer_evidence, build_answer_messages, _messages_bytes
        large = unit("large", "fact" + "\n" * 7900)
        noise = unit("noise", "Small unrelated source.")
        claim = source_claim("A", large)
        stored = {"retained_units": [large, noise], "claims": [claim],
                  "records": [{"record_id": "A", "support_ids": ["A"]}], "relations": [],
                  "vectors": [[1.0] + [0.0] * 1023] * 2}
        ranked = [{"unit_id": "large", "score": 1.0}, {"unit_id": "noise", "score": 0.9}]
        with patch.object(oracle_probe, "_top_raw_ids", return_value=ranked):
            evidence, ranking = _design_answer_evidence("Question?", stored, OfflineEncoder(), lambda text: 10)
        self.assertEqual({u["unit_id"] for u in evidence}, {"noise"})
        self.assertTrue(evidence[0]["coverage_incomplete"])
        self.assertLessEqual(_messages_bytes(build_answer_messages("Question?", None, evidence)), 24576)
        self.assertEqual(ranking[0]["omission_reason"], "required_evidence_exceeds_budget")

    def test_update_groups_split_when_new_claims_alone_exceed_budget(self):
        from dive_memory import build_w2_messages
        from oracle_probe import plan_update_groups, _messages_bytes
        claims = [source_claim(f"C{i}", unit(f"u{i}", "fact" + " " * 5000)) for i in range(4)]
        groups = plan_update_groups(claims)
        self.assertGreater(len(groups), 1)
        self.assertEqual([c["claim_id"] for g in groups for c in g], [c["claim_id"] for c in claims])
        for group in groups:
            mapping = {c["claim_id"]: [x["claim_id"] for x in group[:i]] for i, c in enumerate(group)}
            self.assertLessEqual(_messages_bytes(build_w2_messages(
                group, [], mapping, {cid: False for cid in mapping}, claims=[], relations=[])), 24576)

    def test_real_saved_oracle_rebuild_keeps_ids_without_api_or_key(self):
        from oracle_probe import process_design_units, ReplayMissing
        folder = ROOT / "runs" / "design-validation" / "20261113-053529-752027f0"
        if not folder.exists():
            self.skipTest("Optional local real-response artifact is not present")
        entries = [json.loads(line) for line in (folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
        sources = [u for entry in entries if entry["stage"] == "W1" for u in entry["input"]["units"]]
        before = copy.deepcopy(entries)
        tracked = TrackedModel(None, Path("UNUSED_NO_WRITE"), 16, capture_validation=True,
                               resume_entries=entries, replay_only=True)
        report = {"name": "oracle"}
        with (patch.object(oracle_probe, "read_key", side_effect=AssertionError("No key")),
              patch("dive_minimal.urllib.request.build_opener", side_effect=AssertionError("No network")),
              redirect_stdout(io.StringIO()), self.assertRaises(ReplayMissing)):
            process_design_units(sources, OfflineEncoder(), lambda text: min(len(text) // 4, 512), tracked, report)
        self.assertEqual(entries, before)
        self.assertEqual(len(tracked.replays), 7)
        self.assertEqual(tracked.calls, [])
        self.assertEqual(tracked.stage, "oracle/W2-2")
        self.assertEqual({c["claim_id"] for c in report["claims"]},
                         {f"b0c{i:03d}" for i in range(6)} | {"b1c000", "b1c001"})
        self.assertTrue(all(set(r) == {"record_id", "support_ids"} for r in report["records"]))
        self.assertTrue(any(r["kind"] == "historical_revalidation" for r in tracked.replays))

    def test_resume_refuses_changed_input_and_corrupt_hash_before_api(self):
        from oracle_probe import replay_validation_response
        folder = ROOT / "runs" / "design-validation" / "20261113-053529-752027f0"
        if not folder.exists(): self.skipTest("Optional local real-response artifact is not present")
        entry = json.loads((folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0])
        bad = copy.deepcopy(entry)
        bad["output"]["claims"][0]["value"] = "tampered"
        with self.assertRaises(ValueError): replay_validation_response(bad)
        tracked = TrackedModel(None, Path("UNUSED_NO_WRITE"), 1, resume_entries=[entry], replay_only=True)
        tracked.stage = entry["api_stage"]
        payload = copy.deepcopy(entry["input"])
        payload["units"][0]["messages"][0]["content"] = "different source"
        messages = [{"role": "system", "content": entry["system_prompt"]},
                    {"role": "user", "content": json.dumps(payload)}]
        with self.assertRaises(ModelError): tracked.complete(messages, max_tokens=entry["max_tokens"])
        self.assertEqual(tracked.calls, [])

    def test_reversed_model_claim_order_is_sorted_before_update(self):
        from dive_memory import CLAIM_FIELDS, record_views
        from oracle_probe import process_design_units
        sources = [unit("u0", "I live in Chengdu."), unit("u1", "I now live in Hangzhou.")]
        sources[1]["date"] = "2023-05-01T10:00"

        class Flow(FakeModel):
            def complete(self, messages, *, max_tokens, json_mode=False):
                data = json.loads(messages[-1]["content"])
                if "units" in data:
                    cs = [source_claim("unused", u, city) for u, city in zip(data["units"], ("Chengdu", "Hangzhou"))]
                    response = {"decisions": [{"unit_id": u["unit_id"], "decision": "keep", "status": "complete"} for u in data["units"]],
                                "claims": [{k: c[k] for k in CLAIM_FIELDS} for c in reversed(cs)]}
                else:
                    previous, current = data["new_claims"]
                    if previous["value"] != "Chengdu": raise AssertionError("source order not restored")
                    response = {"operations": [proposal(previous), proposal(current, "CHANGE", [previous["claim_id"]],
                                                                          same_slot=True, value_relation="exclusive")]}
                self.response = json.dumps(response)
                return super().complete(messages, max_tokens=max_tokens, json_mode=json_mode)

        report = process_design_units(sources, OfflineEncoder(), len, Flow(), {"name": "reversed"})
        self.assertEqual({r["value"]: r["state"] for r in record_views(report["records"], report["claims"], report["relations"])},
                         {"Chengdu": "historical", "Hangzhou": "active"})
        self.assertEqual(report["claims"][0]["claim_id"], "b0c001")

    def test_cross_batch_short_reply_uses_context_without_reextracting_it(self):
        from dive_memory import CLAIM_FIELDS
        from oracle_probe import process_design_units
        sources = [unit(f"u{i}", f"Greeting {i}.") for i in range(7)]
        sources += [unit("u7", "Will you continue Atlas tomorrow?"), unit("u8", "Yes.")]
        sources[7]["messages"][0]["role"] = "assistant"
        for i, source in enumerate(sources): source["messages"][0]["message_index"] = i

        class Flow(FakeModel):
            def complete(self, messages, *, max_tokens, json_mode=False):
                data = json.loads(messages[-1]["content"])
                if "units" in data:
                    current = data["units"]
                    answer = next((u for u in current if u["unit_id"] == "u8"), None)
                    cs = []
                    if answer:
                        question = next(u for u in data["context_units"] if u["unit_id"] == "u7")
                        c = source_claim("unused", answer, "Continue Atlas", modality="planned")
                        c["refs"].append(dict(source_claim("unused", question)["refs"][0], purpose="context"))
                        cs = [{k: c[k] for k in CLAIM_FIELDS}]
                    response = {"decisions": [{"unit_id": u["unit_id"], "decision": "keep" if u is answer else "discard", "status": "complete"}
                                              for u in current], "claims": cs}
                else: response = {"operations": [proposal(c) for c in data["new_claims"]]}
                self.response = json.dumps(response)
                return super().complete(messages, max_tokens=max_tokens, json_mode=json_mode)

        report = process_design_units(sources, OfflineEncoder(), len, Flow(), {"name": "context"})
        self.assertEqual({u["unit_id"] for u in report["retained_units"]}, {"u7", "u8"})
        self.assertEqual(report["claims"][0]["speaker"], "user")

    def test_candidate_trimming_counts_full_prompt_and_marks_coverage(self):
        from oracle_probe import fit_design_candidates, _messages_bytes
        from dive_memory import build_w2_messages
        claims = [source_claim(f"C{i}", unit(f"s{i}", "x" * 1200)) for i in range(32)]
        records = [{"record_id": c["claim_id"], "support_ids": [c["claim_id"]]} for c in claims]
        new = source_claim("new", unit("fresh", "A new observation."))
        selected, mapping, coverage = fit_design_candidates(
            [new], records, {"new": [r["record_id"] for r in records]}, claims=claims, relations=[])
        self.assertLess(len(selected), len(records))
        self.assertFalse(coverage["new"])
        self.assertLessEqual(_messages_bytes(build_w2_messages(
            [new], selected, mapping, coverage, claims=claims, relations=[])), 24576)

    def test_actual_update_plan_stops_before_calls_exceed_remaining_budget(self):
        from dive_memory import CLAIM_FIELDS
        from oracle_probe import process_design_units
        source = unit("many", "Several independent details.")
        cs = [source_claim("unused", source, str(i)) for i in range(5)]
        response = {"decisions": [{"unit_id": "many", "decision": "keep", "status": "complete"}],
                    "claims": [{k: c[k] for k in CLAIM_FIELDS} for c in cs]}
        with self.tempdir("call-budget-") as folder:
            tracked = TrackedModel(FakeModel(json.dumps(response)), Path(folder) / "calls.jsonl", 2)
            report = {"name": "budget"}
            with redirect_stdout(io.StringIO()), self.assertRaises(ModelError):
                process_design_units([source], OfflineEncoder(), len, tracked, report)
            self.assertEqual(len(tracked.calls), 1)
            self.assertEqual(report["planned_W2"], 2)
            self.assertEqual(len(report["pending_claim_ids"]), 5)
            self.assertEqual(report["updates"], [])

    def test_semantic_start_selects_remaining_cases_and_records_hard_limit(self):
        class Encoder(OfflineEncoder):
            tokenizer = type("Tokenizer", (), {"encode": staticmethod(list)})()
        with self.tempdir("semantic-start-") as folder:
            with (patch.object(oracle_probe, "ROOT", Path(folder)),
                  patch.object(oracle_probe, "cached_encoder", return_value=Encoder()),
                  patch.object(oracle_probe, "design_retrieval_check", return_value={"passed": True}),
                  patch.object(oracle_probe, "read_key", return_value="OFFLINE_ONLY"),
                  patch.object(oracle_probe, "DeepSeekModel", return_value=FakeModel()),
                  patch.object(oracle_probe, "process_design_units") as process,
                  patch.object(oracle_probe, "persist_design_case"), redirect_stdout(io.StringIO())):
                result = oracle_probe.main(["validate-design", "--validation-mode", "semantic",
                                            "--start", "3", "--max-calls", "9"])
            self.assertEqual(result, 0)
            self.assertEqual([call.args[-1]["name"] for call in process.call_args_list],
                             ["unknown-time", "event-identity", "short-context"])
            path, = (Path(folder) / "runs" / "design-validation").glob("*/summary.json")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["call_plan"]["new_call_hard_limit"], 9)
            self.assertTrue(saved["call_plan"]["W2_planned_after_extraction"])

    def test_cli_replay_only_is_keyless_and_saves_partial_rebuilt_state(self):
        source_folder = ROOT / "runs" / "design-validation" / "20261113-053529-752027f0"
        if not source_folder.exists(): self.skipTest("Optional local real-response artifact is not present")
        entries = [json.loads(line) for line in (source_folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
        units = [u for e in entries if e["stage"] == "W1" for u in e["input"]["units"]]
        class Encoder(OfflineEncoder):
            tokenizer = type("Tokenizer", (), {"encode": staticmethod(lambda text: list(range(min(len(text) // 4, 512))))})()
        with self.tempdir("cli-replay-") as folder:
            with (patch.object(oracle_probe, "ROOT", Path(folder)),
                  patch.object(oracle_probe, "cached_encoder", return_value=Encoder()),
                  patch.object(oracle_probe, "design_retrieval_check", return_value={"passed": True}),
                  patch.object(oracle_probe, "make_probe_units", return_value=units),
                  patch.object(oracle_probe, "read_key", side_effect=AssertionError("No key")),
                  patch.object(oracle_probe, "DeepSeekModel", side_effect=AssertionError("No paid client")),
                  redirect_stdout(io.StringIO())):
                code = oracle_probe.main(["validate-design", "--validation-mode", "oracle",
                                          "--data", str(ROOT / "data" / "longmemeval_oracle.json"),
                                          "--resume-from", str(source_folder), "--replay-only", "--max-calls", "16"])
            self.assertEqual(code, 2)
            path, = (Path(folder) / "runs" / "design-validation").glob("*/summary.json")
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "replay_incomplete")
            self.assertEqual(report["failed_stage"], "oracle/W2-2")
            self.assertEqual(len(report["replays"]), 7)
            self.assertEqual(report["usage"]["attempted_paid_calls"], 0)
            self.assertEqual(len(report["oracle"]["claims"]), 8)
            identity = {"dataset_sha256": report["dataset_sha256"],
                        "item": {"question_id": report["oracle"]["question_id"]}}
            # An offline-only run has no new responses; it still retains its verified ancestors.
            recovered = oracle_probe._load_resume_entries(path.parent, identity, "deepseek-flash")
            self.assertEqual(len(recovered), 7)
            loop = Path(folder) / "cycle"
            loop.mkdir()
            (loop / "summary.json").write_text(json.dumps({
                "mode": "oracle", "model": "deepseek-flash", "dataset_sha256": report["dataset_sha256"],
                "oracle": {"question_id": report["oracle"]["question_id"]}, "resume_source": str(loop),
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                oracle_probe._load_resume_entries(loop, identity, "deepseek-flash")
        with self.assertRaises(ValueError):
            oracle_probe._load_resume_entries(source_folder,
                                             {"dataset_sha256": "different", "item": {"question_id": "other"}},
                                             "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
