"""Offline checks of the design-validation subset; no model or network calls."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dive_memory import (apply_updates, extract_claims, judge_updates, record_views,
                         current_conflict_ids, validate_memory)
from dive_minimal import DeepSeekModel, ModelError
from oracle_probe import (TrackedModel, persist_design_case, process_design_units,
                          usage_summary)


class FakeModel:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def complete(self, messages, *, max_tokens, json_mode=False):
        self.calls.append(copy.deepcopy(messages))
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload, ensure_ascii=False), {
            "provider": "offline-fake", "model": "offline-fake",
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        }


def unit(uid="u0", text="I live in Chengdu.", role="user", start=0,
         date="2026-09-01T10:00:00"):
    return {"unit_id": uid, "session_id": "s0", "date": date, "messages": [{
        "role": role, "content": text, "message_index": 0,
        "start": start, "end": start + len(text),
    }]}


def reference(source):
    message = source["messages"][0]
    return {"unit_id": source["unit_id"], "message_index": message["message_index"],
            "start": message["start"], "end": message["end"],
            "quote": message["content"], "purpose": "claim"}


def claim_payload(source, value="Chengdu", **overrides):
    result = {"unit_id": source["unit_id"], "kind": "state", "subject": "user",
              "attribute": "residence", "value": value, "scope": "primary residence",
              "modality": "actual", "polarity": "positive", "temporal_mode": "current",
              "time_lower": None, "time_upper": None, "time_expression": None,
              "refs": [reference(source)]}
    result.update(overrides)
    return result


def claim(cid, source, value="Chengdu", **overrides):
    result = claim_payload(source, value, **overrides)
    result.update(claim_id=cid, observed_at=source["date"], speaker=source["messages"][0]["role"])
    return result


def operation(new, action="ADD", old_ids=None, **overrides):
    result = {"claim_id": new["claim_id"], "action": action, "old_ids": old_ids or [],
              "same_slot": None, "value_relation": "unknown", "older_id": None,
              "newer_id": None, "basis_refs": copy.deepcopy(new["refs"]),
              "reason": "Directly supported by the provided source.", "same_event": None}
    result.update(overrides)
    return result


class ExtractionTests(unittest.TestCase):
    def test_quote_uses_original_message_offsets_and_derived_speaker(self):
        source = unit(text="Consider moving to Beijing.", role="assistant", start=37)
        payload = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                   "claims": [claim_payload(source, "Beijing", modality="suggested")]}
        result = extract_claims([source], FakeModel(payload), id_prefix="check")
        extracted, = result["claims"]
        self.assertEqual(extracted["refs"][0]["start"], 37)
        self.assertEqual(extracted["speaker"], "assistant")
        self.assertEqual(extracted["modality"], "suggested")
        self.assertEqual(extracted["observed_at"], source["date"])

    def test_invented_source_or_altered_quote_is_rejected(self):
        source = unit(start=12)
        good = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                "claims": [claim_payload(source)]}
        unknown = copy.deepcopy(good)
        unknown["claims"][0]["refs"][0]["unit_id"] = "not-supplied"
        changed = copy.deepcopy(good)
        changed["claims"][0]["refs"][0]["quote"] = "I live in Hangzhou."
        for payload in (unknown, changed):
            with self.subTest(payload=payload), self.assertRaises(ModelError):
                extract_claims([source], FakeModel(payload))

    def test_raw_only_cannot_silently_apply_a_partial_extraction(self):
        source = unit()
        for status in ("raw_only", "overflow"):
            payload = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": status}],
                       "claims": [claim_payload(source)]}
            with self.subTest(status=status), self.assertRaises(ModelError):
                extract_claims([source], FakeModel(payload))
        payload["claims"] = []
        self.assertEqual(extract_claims([source], FakeModel(payload))["claims"], [])

class UsageFailureTests(unittest.TestCase):
    def test_provider_usage_survives_output_limit_and_empty_content(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        for finish, content in (("length", "{}"), ("stop", "")):
            response = {"model": "offline", "usage": usage,
                        "choices": [{"finish_reason": finish, "message": {"content": content}}]}
            with self.subTest(finish=finish), patch("dive_minimal.urllib.request.build_opener") as build:
                build.return_value.open.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(response).encode("utf-8"))
                with self.assertRaises(ModelError) as raised:
                    DeepSeekModel("DUMMY_ONLY").complete([], max_tokens=20)
                self.assertEqual(raised.exception.call_info["usage"], usage)

    def test_tracked_failure_keeps_known_usage_and_excludes_credentials(self):
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        info = {"provider": "offline", "model": "offline",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "authorization": "NEVER_LOG_A_CREDENTIAL"}
        error = ModelError("Offline incomplete output", call_info=info)
        with tempfile.TemporaryDirectory(prefix="usage-", dir=parent) as folder:
            path = Path(folder) / "calls.jsonl"
            model = TrackedModel(FakeModel(error=error), path, max_calls=1)
            with patch("sys.stdout", new=io.StringIO()), self.assertRaises(ModelError):
                model.complete([], max_tokens=20)
            summary = usage_summary(model.calls)
            self.assertEqual(summary["known_usage_totals"]["total_tokens"], 120)
            self.assertEqual(summary["calls_with_unknown_usage"], 0)
            self.assertNotIn("NEVER_LOG_A_CREDENTIAL", path.read_text(encoding="utf-8"))

class LocalModeTests(unittest.TestCase):
    def test_default_validation_mode_never_reads_a_key_or_creates_a_paid_client(self):
        import oracle_probe

        class Encoder:
            tokenizer = type("Tokenizer", (), {"encode": staticmethod(list)})()

        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="local-mode-", dir=parent) as folder:
            with (patch.object(oracle_probe, "ROOT", Path(folder)),
                  patch.object(oracle_probe, "cached_encoder", return_value=Encoder()),
                  patch.object(oracle_probe, "design_retrieval_check",
                               return_value={"passed": True, "paid_api_calls": 0}),
                  patch.object(oracle_probe, "read_key") as key,
                  patch.object(oracle_probe, "DeepSeekModel") as client,
                  patch("sys.stdout", new=io.StringIO())):
                code = oracle_probe.main(["validate-design"])
            self.assertEqual(code, 0)
            key.assert_not_called()
            client.assert_not_called()
            reports = list((Path(folder) / "runs" / "design-validation").glob("*/summary.json"))
            self.assertEqual(len(reports), 1)
            saved = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["usage"]["attempted_paid_calls"], 0)
            self.assertEqual(saved["status"], "local_retrieval_completed")

class ContractBoundaryTests(unittest.TestCase):
    def payload(self):
        source = unit()
        return source, {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                        "claims": [claim_payload(source)]}

    def assert_bad_claim(self, field, value):
        source, payload = self.payload()
        payload["claims"][0][field] = value
        with self.assertRaises(ModelError) as raised:
            extract_claims([source], FakeModel(payload))
        self.assertIn(field, str(raised.exception))
        self.assertEqual(raised.exception.call_info["usage"]["total_tokens"], 24)

    def test_text_types_empty_policy_and_length_boundaries(self):
        limits = {"subject": 128, "attribute": 64, "value": 512, "scope": 256}
        for field, maximum in limits.items():
            for value in (None, 42, [], {}, "x" * (maximum + 1)):
                with self.subTest(field=field, value_type=type(value).__name__):
                    self.assert_bad_claim(field, value)
            source, payload = self.payload()
            payload["claims"][0][field] = "x" * maximum
            self.assertEqual(extract_claims([source], FakeModel(payload))["claims"][0][field],
                             "x" * maximum)
            if field != "scope":
                for value in ("", "   "):
                    with self.subTest(field=field, empty=value):
                        self.assert_bad_claim(field, value)
        source, payload = self.payload()
        payload["claims"][0]["scope"] = ""
        self.assertEqual(extract_claims([source], FakeModel(payload))["claims"][0]["scope"], "")

    def test_required_fields_and_enum_types(self):
        from dive_memory import CLAIM_FIELDS
        for field in CLAIM_FIELDS:
            source, payload = self.payload()
            del payload["claims"][0][field]
            with self.subTest(missing=field), self.assertRaises(ModelError) as raised:
                extract_claims([source], FakeModel(payload))
            self.assertIn(field, str(raised.exception))
        for field in ("kind", "modality", "polarity", "temporal_mode"):
            for value in (None, [], "INVALID", "Actual"):
                with self.subTest(field=field, value=value):
                    self.assert_bad_claim(field, value)

    def test_time_null_iso_and_expression_boundaries(self):
        for field in ("time_lower", "time_upper"):
            for value in (None, "2026-10-01", "2026-10-01T10:00:00"):
                source, payload = self.payload()
                payload["claims"][0][field] = value
                extract_claims([source], FakeModel(payload))
            for value in (2026, [], "", "2026-02-30", "yesterday"):
                with self.subTest(field=field, value=value):
                    self.assert_bad_claim(field, value)
        for value in (None, "x" * 128):
            source, payload = self.payload()
            payload["claims"][0]["time_expression"] = value
            extract_claims([source], FakeModel(payload))
        self.assert_bad_claim("time_expression", "x" * 129)
        source, payload = self.payload()
        payload["claims"][0].update(time_lower="2026-10-02", time_upper="2026-10-01")
        with self.assertRaises(ModelError):
            extract_claims([source], FakeModel(payload))

    def test_reference_fields_counts_and_positions(self):
        source, payload = self.payload()
        good = payload["claims"][0]["refs"][0]
        bad_refs = [[], [copy.deepcopy(good) for _ in range(5)]]
        for field in good:
            altered = copy.deepcopy(good)
            del altered[field]
            bad_refs.append([altered])
        for field, value in (("start", True), ("start", -1), ("end", -1),
                             ("message_index", False), ("quote", "changed")):
            altered = copy.deepcopy(good)
            altered[field] = value
            bad_refs.append([altered])
        bad_refs.append([dict(good, unknown="extra")])
        for refs in bad_refs:
            with self.subTest(refs=refs):
                self.assert_bad_claim("refs", refs)

    def test_w1_example_and_schema_are_valid_json(self):
        from dive_memory import W1_RESPONSE_SCHEMA, W2_RESPONSE_SCHEMA
        source, payload = self.payload()
        model = FakeModel(payload)
        extract_claims([source], model)
        prompt = model.calls[0][0]["content"]
        example = json.loads(prompt.split("JSON format example: ", 1)[1].split("\n\n", 1)[0])
        self.assertIsInstance(example["claims"][0]["scope"], str)
        self.assertIn("256", prompt)
        self.assertIn('"scope"', json.dumps(W1_RESPONSE_SCHEMA))
        self.assertIn('"same_slot"', json.dumps(W2_RESPONSE_SCHEMA))

    def test_w2_required_fields_reason_and_container_types(self):
        new = claim("new", unit("u1"))
        valid = operation(new)
        variants = []
        for field in valid.keys() - {"older_id", "newer_id", "target_claim_ids"}:
            altered = copy.deepcopy(valid)
            del altered[field]
            variants.append((field, altered))
        for field, value in (("reason", ""), ("reason", " " * 3), ("reason", "x" * 2001),
                             ("old_ids", "old"), ("old_ids", [True]),
                             ("value_relation", []), ("value_relation", "EXCLUSIVE")):
            variants.append((field, dict(valid, **{field: value})))
        for field, proposal in variants:
            with self.subTest(field=field), self.assertRaises(ModelError) as raised:
                judge_updates([new], [], {"new": []}, FakeModel({"operations": [proposal]}))
            self.assertIn(field, str(raised.exception))
            self.assertEqual(raised.exception.call_info["usage"]["total_tokens"], 24)

    def test_saved_successful_w1_can_be_revalidated_without_api(self):
        from dive_memory import CLAIM_FIELDS
        from oracle_probe import design_cases
        path = ROOT / "runs" / "design-validation" / "20261111-011831-28f9243e" / "summary.json"
        if not path.exists():
            self.skipTest("Optional local successful W1 artifact is not present")
        saved = json.loads(path.read_text(encoding="utf-8"))["cases"][0]["extraction"][0]
        payload = {"decisions": saved["decisions"],
                   "claims": [{key: c[key] for key in CLAIM_FIELDS} for c in saved["claims"]]}
        units = design_cases()[0]["units"]
        with patch("dive_minimal.urllib.request.build_opener", side_effect=AssertionError("No network")):
            result = extract_claims(units, FakeModel(payload))
        for actual, previous in zip(result["claims"], saved["claims"]):
            self.assertEqual({key: actual[key] for key in previous}, previous)
        self.assertEqual(len(result["claims"]), len(saved["claims"]))

class ResponseReplayTests(unittest.TestCase):
    def capture(self, payload, folder):
        base = FakeModel(payload)
        base._key = "PRIVATE_KEY_CANARY"
        tracked = TrackedModel(base, Path(folder) / "calls.jsonl", 1, capture_validation=True)
        source = unit()
        error = None
        with patch("sys.stdout", new=io.StringIO()):
            try:
                extract_claims([source], tracked)
            except ModelError as caught:
                error = caught
        path = Path(folder) / "responses.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return rows[-1], error, path.read_text(encoding="utf-8")

    def test_invalid_known_field_is_recorded_before_validation_and_replays_offline(self):
        from oracle_probe import replay_validation_response
        source = unit()
        payload = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                   "claims": [claim_payload(source, scope=None)]}
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="replay-bad-", dir=parent) as folder:
            entry, error, serialized = self.capture(payload, folder)
        self.assertIsInstance(error, ModelError)
        self.assertTrue(entry["replayable"])
        self.assertIsNone(entry["output"]["claims"][0]["scope"])
        self.assertNotIn("PRIVATE_KEY_CANARY", serialized)
        for key in ("request_hash", "prompt_hash", "output_hash", "schema_hash"):
            self.assertEqual(len(entry[key]), 64)
        self.assertEqual(entry["call_info"]["usage"]["total_tokens"], 24)
        with (patch("dive_minimal.urllib.request.build_opener", side_effect=AssertionError("No API")),
              self.assertRaises(ModelError) as replayed):
            replay_validation_response(entry)
        self.assertIn("scope", str(replayed.exception))

    def test_valid_response_replays_and_modified_hash_is_rejected(self):
        from oracle_probe import replay_validation_response
        source = unit()
        payload = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                   "claims": [claim_payload(source)]}
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="replay-good-", dir=parent) as folder:
            entry, error, _ = self.capture(payload, folder)
        self.assertIsNone(error)
        self.assertEqual(len(replay_validation_response(entry)["claims"]), 1)
        altered = copy.deepcopy(entry)
        altered["input"]["units"][0]["messages"][0]["content"] = "changed"
        with self.assertRaises(ValueError):
            replay_validation_response(altered)

    def test_unknown_sensitive_fields_and_oversized_output_are_not_claimed_replayable(self):
        source = unit()
        good = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                "claims": [claim_payload(source)]}
        unknown = copy.deepcopy(good)
        unknown["api_key"] = "RESPONSE_SECRET_CANARY"
        oversized = copy.deepcopy(good)
        oversized["claims"][0]["scope"] = "x" * 180_000
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        for payload in (unknown, oversized):
            with self.subTest(oversized=payload is oversized):
                with tempfile.TemporaryDirectory(prefix="replay-limit-", dir=parent) as folder:
                    entry, _, serialized = self.capture(payload, folder)
                self.assertFalse(entry["replayable"])
                self.assertNotIn("RESPONSE_SECRET_CANARY", serialized)
                self.assertLessEqual(len(serialized.encode("utf-8")), 131_073)

class GroundedReferenceRegressionTests(unittest.TestCase):
    def extract(self, source, refs):
        payload = {"decisions": [{"unit_id": source["unit_id"], "decision": "keep", "status": "complete"}],
                   "claims": [claim_payload(source, refs=refs)]}
        before = copy.deepcopy(payload)
        result = extract_claims([source], FakeModel(payload))
        self.assertEqual(payload, before)
        return result

    def test_unique_quote_is_located_in_original_unicode_fragment(self):
        source = unit(text="🙂前文。我现在住杭州。", start=37)
        quote = "我现在住杭州。"
        expected = 37 + source["messages"][0]["content"].index(quote)
        for start, end in ((None, None), (37, 38), (expected, None)):
            with self.subTest(start=start, end=end):
                result = self.extract(source, [dict(reference(source), quote=quote, start=start, end=end)])
                ref, = result["claims"][0]["refs"]
                self.assertEqual((ref["start"], ref["end"]), (expected, expected + len(quote)))
                if (start is not None and start != expected) or (end is not None and end != expected + len(quote)):
                    self.assertTrue(result["warnings"])

    def test_repeated_quote_needs_a_valid_unambiguous_supplied_span(self):
        source = unit(text="杭州，杭州。", start=9)
        good = dict(reference(source), quote="杭州", start=12, end=14)
        self.assertEqual(self.extract(source, [good])["claims"][0]["refs"][0], good)
        for ref in (dict(good, start=None, end=None), dict(good, start=9, end=12)):
            with self.subTest(ref=ref), self.assertRaises(ModelError):
                self.extract(source, [ref])

    def test_alignment_never_guesses_other_messages_or_altered_text(self):
        source = unit(text="我住杭州。")
        other = dict(source["messages"][0], message_index=1, content="我住成都。")
        source["messages"].append(other)
        base = dict(reference(source), start=None, end=None)
        for ref in (dict(base, message_index=1), dict(base, unit_id="missing"),
                    dict(base, quote="我住 杭州。"), dict(base, message_index=99)):
            with self.subTest(ref=ref), self.assertRaises(ModelError):
                self.extract(source, [ref])

    def test_relationship_basis_is_canonicalized_only_to_permitted_refs(self):
        new = claim("new", unit("u1"))
        for offsets in ((None, None), (1, 2)):
            proposal = operation(new, basis_refs=[dict(new["refs"][0], start=offsets[0], end=offsets[1])])
            before = copy.deepcopy(proposal)
            checked = judge_updates([new], [], {"new": []}, FakeModel({"operations": [proposal]}))
            self.assertEqual(checked["proposals"][0]["basis_refs"], new["refs"])
            self.assertEqual(proposal, before)
        repeated = copy.deepcopy(new)
        repeated["refs"].append(dict(repeated["refs"][0], start=50, end=50 + len(repeated["refs"][0]["quote"])))
        proposal = operation(repeated, basis_refs=[dict(new["refs"][0], start=None, end=None)])
        with self.assertRaises(ModelError):
            judge_updates([repeated], [], {"new": []}, FakeModel({"operations": [proposal]}))
        for field, value in (("purpose", "time"), ("message_index", 99), ("quote", "Invented")):
            proposal = operation(new, basis_refs=[dict(new["refs"][0], **{field: value})])
            with self.subTest(field=field), self.assertRaises(ModelError):
                judge_updates([new], [], {"new": []}, FakeModel({"operations": [proposal]}))


class Memory:
    """Test fixture using exactly the production persisted form."""
    def __init__(self, *sources):
        self.claims = [dict(copy.deepcopy(source), validity="valid") for source in sources]
        self.records = [{"record_id": source["claim_id"], "support_ids": [source["claim_id"]]}
                        for source in sources]
        self.relations = []

    def view(self):
        return {row["record_id"]: row for row in record_views(self.records, self.claims, self.relations)}

    def run(self, new_claims, proposals, candidates=None, coverage=None):
        candidates = candidates or {c["claim_id"]: [r["record_id"] for r in self.records]
                                    for c in new_claims}
        judged = judge_updates(new_claims, self.records, candidates,
                               FakeModel({"operations": proposals}), coverage,
                               claims=self.claims, relations=self.relations)
        applied = apply_updates(self.records, new_claims, judged["proposals"],
                                claims=self.claims, relations=self.relations)
        self.records, self.claims = applied["records"], applied["claims"]
        self.relations.extend(applied["relations"])
        return judged


class UnifiedUpdateTests(unittest.TestCase):
    def setUp(self):
        self.old = claim("old", unit())
        self.new = claim("new", unit("u1", "I moved to Hangzhou.", date="2026-10-02T10:00:00"),
                         "Hangzhou", time_lower="2026-10-01", time_upper="2026-10-01")

    def change(self, new=None, old_id="old", **kwargs):
        new = new or self.new
        return operation(new, "CHANGE", [old_id], same_slot=True,
                         value_relation="exclusive", **kwargs)

    def test_storage_has_one_source_of_truth_and_support_preserves_ids(self):
        memory = Memory(self.old)
        repeated = claim("mention", unit("u2", "I still live in Chengdu.", date="2026-09-10"))
        memory.run([repeated], [operation(repeated, "SUPPORT", ["old"], same_slot=True)])
        self.assertEqual(memory.records, [{"record_id": "old", "support_ids": ["old", "mention"]}])
        self.assertEqual([c["claim_id"] for c in memory.claims], ["old", "mention"])
        self.assertEqual(len(memory.view()["old"]["supporting_claims"]), 2)
        validate_memory(memory.records, memory.claims)
        for record in memory.records:
            self.assertEqual(set(record), {"record_id", "support_ids"})

    def test_unknown_candidate_future_id_duplicate_id_and_fabricated_quote_rejected(self):
        cases = [
            ({"new": []}, self.change()),
            ({"new": ["future"]}, self.change()),
            ({"new": ["old", "old"]}, self.change()),
            ({"new": ["old"]}, dict(self.change(), basis_refs=[dict(self.new["refs"][0], quote="invented")])),
        ]
        for mapping, proposal in cases:
            with self.subTest(mapping=mapping, proposal=proposal), self.assertRaises(ModelError):
                judge_updates([self.new], Memory(self.old).records, mapping,
                              FakeModel({"operations": [proposal]}), claims=[self.old])
        with self.assertRaises(ModelError):
            judge_updates([self.old], Memory(self.old).records, {"old": []},
                          FakeModel({}), claims=[self.old])

    def test_false_support_flags_preserve_distinct_pending_information(self):
        for field, value in (("same_slot", False), ("same_event", False)):
            memory = Memory(dict(self.old, kind="event"))
            new = dict(self.new, kind="event")
            proposal = operation(new, "SUPPORT", ["old"], same_slot=True, same_event=True)
            proposal[field] = value
            result = memory.run([new], [proposal])
            with self.subTest(field=field):
                self.assertEqual(result["proposals"][0]["action"], "UNCERTAIN")
                self.assertEqual(result["proposals"][0]["original_proposal"], proposal)
                self.assertEqual(memory.view()["old"]["state"], "active")
                self.assertEqual(memory.view()["new"]["state"], "pending")
                self.assertTrue(all(c["validity"] == "valid" for c in memory.claims))

    def test_event_identity_merges_mentions_without_merging_distinct_events(self):
        race = claim("A", unit(text="I attended race A.", date="2026-10-01"), "race A", kind="event",
                     temporal_mode="dated", time_lower="2026-09-30", time_upper="2026-09-30")
        memory = Memory(race)
        repeat = claim("mentionA", unit("a2", "The same race A.", date="2026-10-02"),
                       "race A", kind="event", temporal_mode="dated",
                       time_lower="2026-09-30", time_upper="2026-09-30")
        other = claim("B", unit("b", "I attended race B.", date="2026-10-08"),
                      "race B", kind="event")
        memory.run([repeat, other],
                   [operation(repeat, "SUPPORT", ["A"], same_slot=True, same_event=True),
                    operation(other)], {"mentionA": ["A"], "B": ["A", "mentionA"]})
        self.assertEqual(len(memory.records), 2)
        self.assertEqual(memory.records[0]["support_ids"], ["A", "mentionA"])

    def test_change_direction_is_computed_and_late_history_does_not_replace_current(self):
        memory = Memory(self.old)
        # Saved model direction is deliberately reversed; source times are authoritative.
        judged = memory.run([self.new], [self.change(older_id="new", newer_id="old")])
        self.assertEqual((judged["proposals"][0]["older_id"], judged["proposals"][0]["newer_id"]),
                         ("old", "new"))
        self.assertEqual(memory.view()["old"]["state"], "historical")
        late = claim("late", unit("late", "I lived in Chengdu in August.", date="2026-11-01"),
                     temporal_mode="dated", time_lower="2026-08-01", time_upper="2026-08-01")
        memory.run([late], [self.change(late, "new")], {"late": ["new"]})
        self.assertEqual(memory.view()["new"]["state"], "active")
        self.assertEqual(memory.view()["late"]["state"], "historical")

    def test_plans_cannot_close_actual_state_and_incomplete_candidates_defer(self):
        for new, coverage in ((dict(self.new, modality="planned"), None),
                              (self.new, {"new": False})):
            memory = Memory(self.old)
            result = memory.run([new], [self.change(new)], coverage=coverage)
            self.assertEqual(result["proposals"][0]["action"], "UNCERTAIN")
            self.assertEqual(memory.view()["old"]["state"], "active")

    def test_missing_overlapping_and_incomparable_times_preserve_uncertainty(self):
        examples = [(None, None), ("2026-10-01", "2026-10-01"),
                    ("2026-10-01T10:00", "2026-10-01T10:00:30"),
                    ("2026-10-01T10:00:00+08:00", "2026-10-02T10:00:00")]
        for first, second in examples:
            memory = Memory(claim("old", unit(date=first)))
            new = claim("new", unit("u1", date=second), "Hangzhou")
            with self.subTest(first=first, second=second):
                result = memory.run([new], [self.change(new)])
                self.assertEqual(result["proposals"][0]["action"], "UNCERTAIN")
                self.assertEqual(memory.view()["old"]["state"], "active")

    def test_same_session_order_requires_unique_current_message_positions(self):
        old = claim("old", unit(date=None), source_session_id="s", source_message_index=1)
        new = claim("new", unit("u1", date=None), "Hangzhou",
                    source_session_id="s", source_message_index=2)
        memory = Memory(old)
        self.assertEqual(memory.run([new], [self.change(new)])["proposals"][0]["action"], "CHANGE")
        for overrides in ({"source_session_id": "another"}, {"source_message_index": 1},
                          {"source_message_index": None}, {"temporal_mode": "dated"},
                          {"time_lower": "2026-10-01"}):
            memory = Memory(old)
            altered = dict(new, **overrides)
            self.assertEqual(memory.run([altered], [self.change(altered)])["proposals"][0]["action"],
                             "UNCERTAIN")

    def test_multiple_primary_quotes_do_not_create_a_false_single_message_order(self):
        source = unit(date=None)
        source["messages"][0]["message_index"] = 1
        source["messages"].append(dict(source["messages"][0], message_index=5))
        payload = {"decisions": [{"unit_id": "u0", "decision": "keep", "status": "complete"}],
                   "claims": [claim_payload(source, refs=[reference(source),
                                 dict(reference(source), message_index=5)])]}
        extracted = extract_claims([source], FakeModel(payload))["claims"][0]
        self.assertIsNone(extracted["source_message_index"])

    def test_strict_time_format_rejects_unhandled_precision_before_model(self):
        from dive_memory import _date, _time_bound
        for value in ("20261001", "2026-10-01T1000", "2026-W40-4", "2026-10-01 10:00",
                      "2026-10-01T10:00:00.1234567", "2026-02-30"):
            with self.subTest(value=value), self.assertRaises(ModelError):
                _date(value)
        for value in ("2026-10-01", "2026-10-01T10:00", "2026-10-01T10:00:00Z",
                      "2026-10-01T10:00:00.123456+08:00"):
            self.assertIsNotNone(_date(value))
        self.assertEqual(_time_bound("9999-12-31", upper=True).year, 9999)

    def test_all_support_observations_contribute_to_time_direction(self):
        memory = Memory(claim("old", unit(date="2026-09-01")))
        support = claim("mention", unit("m", date="2026-09-10"))
        memory.run([support], [operation(support, "SUPPORT", ["old"], same_slot=True)])
        middle = claim("new", unit("mid", date="2026-09-05"), "Hangzhou")
        self.assertEqual(memory.run([middle], [self.change(middle)])["proposals"][0]["action"], "UNCERTAIN")

    def test_partial_correction_invalidates_only_the_targeted_source(self):
        memory = Memory(claim("old", unit(date="2026-09-01")))
        support = claim("mention", unit("m", date="2026-09-10"))
        memory.run([support], [operation(support, "SUPPORT", ["old"], same_slot=True)])
        correction = claim("correction", unit("c", "My September 1 statement was wrong.",
                           date="2026-09-11"), "Hangzhou", temporal_mode="dated",
                           time_lower="2026-09-01", time_upper="2026-09-01")
        memory.run([correction], [operation(correction, "CORRECT", ["old"], same_slot=True,
                    value_relation="exclusive", target_claim_ids=["old"])])
        self.assertEqual({c["claim_id"]: c["validity"] for c in memory.claims},
                         {"old": "corrected", "mention": "valid", "correction": "valid"})
        view = memory.view()["old"]
        self.assertEqual(view["state"], "active")
        self.assertTrue(view["computed_time_lower"].startswith("2026-09-10"))
        self.assertEqual(view["support_ids"], ["old", "mention"])

    def test_ambiguous_multiple_source_correction_is_deferred(self):
        memory = Memory(self.old)
        support = claim("mention", unit("m"))
        memory.run([support], [operation(support, "SUPPORT", ["old"], same_slot=True)])
        result = memory.run([self.new], [operation(self.new, "CORRECT", ["old"], same_slot=True)])
        self.assertEqual(result["proposals"][0]["action"], "UNCERTAIN")
        self.assertTrue(all(c["validity"] == "valid" for c in memory.claims))

    def test_unrelated_nonactual_and_old_evidence_only_corrections_cannot_close(self):
        for overrides, proposal_overrides in (
                ({"attribute": "preference", "value": "tea"}, {"same_slot": False, "value_relation": "additive"}),
                ({"modality": "planned"}, {"same_slot": True}),
                ({}, {"same_slot": True, "basis_refs": self.old["refs"]})):
            memory = Memory(self.old)
            new = dict(self.new, **overrides)
            result = memory.run([new], [operation(new, "CORRECT", ["old"], **proposal_overrides)])
            self.assertEqual(result["proposals"][0]["action"], "UNCERTAIN")
            self.assertEqual(memory.claims[0]["validity"], "valid")

    def test_unknown_target_claim_ids_are_rejected(self):
        memory = Memory(self.old)
        with self.assertRaises(ModelError):
            memory.run([self.new], [operation(self.new, "CORRECT", ["old"], same_slot=True,
                                             target_claim_ids=["unprovided"])])

    def test_explicit_retraction_closes_source_without_inventing_replacement(self):
        memory = Memory(self.old)
        retract = claim("withdrawal", unit("r", "I retract that residence statement."),
                        "retract residence statement", kind="event")
        memory.run([retract], [operation(retract, "RETRACT", ["old"], target_claim_ids=["old"])])
        self.assertEqual(memory.claims[0]["validity"], "retracted")
        self.assertEqual(memory.view()["old"]["state"], "retracted")
        self.assertEqual(memory.view()["withdrawal"]["kind"], "event")

    def test_event_detail_correction_requires_same_event_confirmation(self):
        original = claim("old", unit(text="Race A was on September 30."), "race A Sep30", kind="event")
        corrected = claim("new", unit("new", "I meant October 1 for that race."),
                          "race A Oct1", kind="event")
        for same_event, expected in ((False, "UNCERTAIN"), (True, "CORRECT")):
            memory = Memory(original)
            result = memory.run([corrected], [operation(corrected, "CORRECT", ["old"],
                          same_slot=True, same_event=same_event, value_relation="exclusive",
                          target_claim_ids=["old"])])
            self.assertEqual(result["proposals"][0]["action"], expected)

    def test_support_alias_in_batch_keeps_original_sources_and_exact_target(self):
        memory = Memory(self.old)
        support = claim("mention", unit("m", date="2026-09-02"))
        correction = self.new
        result = memory.run([support, correction], [
            operation(support, "SUPPORT", ["old"], same_slot=True),
            operation(correction, "CORRECT", ["mention"], same_slot=True,
                      target_claim_ids=["old"], basis_refs=self.old["refs"] + correction["refs"])],
            {"mention": ["old"], "new": ["mention"]})
        self.assertEqual(result["proposals"][1]["action"], "CORRECT")
        self.assertEqual(memory.records[0]["support_ids"], ["old", "mention"])
        self.assertEqual(memory.claims[0]["validity"], "corrected")
        self.assertEqual(memory.claims[1]["validity"], "valid")

    def test_historical_support_does_not_hide_a_later_current_observation(self):
        memory = Memory(claim("old", unit(date="2026-09-01")))
        newer = claim("new", unit("n", date="2026-09-05"), "Hangzhou")
        memory.run([newer], [self.change(newer)])
        returned = claim("return", unit("r", "I now live in Chengdu again.", date="2026-09-10"))
        judged = memory.run([returned], [operation(returned, "SUPPORT", ["old"], same_slot=True)])
        self.assertEqual(judged["proposals"][0]["action"], "UNCERTAIN")
        self.assertEqual(memory.view()["old"]["state"], "historical")
        self.assertEqual(memory.view()["return"]["state"], "pending")

    def test_wrong_boolean_types_remain_structural_errors_with_usage(self):
        for field in ("same_slot", "same_event"):
            for value in ("true", "null", 0, 1, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ModelError) as raised:
                    judge_updates([self.new], [], {"new": []},
                                  FakeModel({"operations": [operation(self.new, **{field: value})]}))
                self.assertIn(field, str(raised.exception))
                self.assertEqual(raised.exception.call_info["usage"]["total_tokens"], 24)

    def test_inputs_are_not_mutated_by_judging_or_applying(self):
        memory = Memory(self.old)
        proposals = [self.change()]
        before = copy.deepcopy((memory.records, memory.claims, memory.relations, self.new, proposals))
        judged = judge_updates([self.new], memory.records, {"new": ["old"]},
                               FakeModel({"operations": proposals}), claims=memory.claims,
                               relations=memory.relations)
        apply_updates(memory.records, [self.new], judged["proposals"],
                      claims=memory.claims, relations=memory.relations)
        self.assertEqual((memory.records, memory.claims, memory.relations, self.new, proposals), before)


class ConflictPropagationTests(unittest.TestCase):
    def memory(self):
        a = claim("A", unit("a", "I live in Suzhou.", date=None), "Suzhou")
        b = claim("B", unit("b", "I live in Nanjing.", date="2026-10-01"), "Nanjing")
        memory = Memory(a)
        memory.run([b], [operation(b, "UNCERTAIN", ["A"], same_slot=True,
                                  value_relation="exclusive")])
        return memory

    def test_conflict_follows_change_then_correction_of_historical_source(self):
        memory = self.memory()
        later = claim("C", unit("c", date="2026-11-01"), "Hangzhou")
        memory.run([later], [operation(later, "CHANGE", ["B"], same_slot=True,
                                      value_relation="exclusive")], {"C": ["B"]})
        self.assertEqual(current_conflict_ids(memory.records, memory.relations, memory.claims), {"A", "C"})
        correction = claim("D", unit("d", date="2026-11-02"), "Chengdu",
                           temporal_mode="dated", time_lower="2026-10-01", time_upper="2026-10-01")
        memory.run([correction], [operation(correction, "CORRECT", ["B"], same_slot=True,
                          target_claim_ids=["B"])], {"D": ["B"]})
        conflicts = current_conflict_ids(memory.records, memory.relations, memory.claims)
        self.assertTrue({"A", "C"} <= conflicts)

    def test_conflict_follows_actual_replacement_correction_but_not_pure_retraction(self):
        for action in ("CORRECT", "RETRACT"):
            memory = self.memory()
            corrected = claim("C", unit("c", date=None), "Hangzhou" if action == "CORRECT" else "withdrawal",
                              kind="state" if action == "CORRECT" else "event")
            memory.run([corrected], [operation(corrected, action, ["B"], same_slot=True,
                           target_claim_ids=["B"], value_relation="exclusive")], {"C": ["B"]})
            expected = {"A", "C"} if action == "CORRECT" else set()
            self.assertEqual(current_conflict_ids(memory.records, memory.relations, memory.claims), expected)


class CanonicalFlowTests(unittest.TestCase):
    class Array:
        def __init__(self, values):
            self.values = values
        def tolist(self):
            return copy.deepcopy(self.values)

    class Encoder:
        def encode(self, texts, **kwargs):
            return CanonicalFlowTests.Array([[1.0] + [0.0] * 1023 for _ in texts])

    def test_w1_w2_persistence_reopens_with_same_canonical_ids_and_states(self):
        class FlowModel(FakeModel):
            def complete(self, messages, *, max_tokens, json_mode=False):
                data = json.loads(messages[-1]["content"])
                if "units" in data:
                    self.payload = {"decisions": [{"unit_id": u["unit_id"], "decision": "keep",
                                                  "status": "complete"} for u in data["units"]],
                                    "claims": [claim_payload(data["units"][0]),
                                               claim_payload(data["units"][1], "Hangzhou")]}
                else:
                    first, last = data["new_claims"]
                    self.payload = {"operations": [operation(first), operation(last, "CHANGE",
                        [first["claim_id"]], same_slot=True, value_relation="exclusive")]}
                return super().complete(messages, max_tokens=max_tokens, json_mode=json_mode)
        sources = [unit(), unit("u1", "I moved to Hangzhou.", date="2026-10-01")]
        processed = process_design_units(sources, self.Encoder(), len, FlowModel(), {"name": "canonical"})
        parent = ROOT / "runs" / "design-validation" / "offline-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="canonical-", dir=parent) as folder:
            restored = persist_design_case(Path(folder) / "memory.sqlite3", "canonical", processed)
        self.assertEqual(restored["claims"], processed["claims"])
        self.assertEqual(restored["records"], processed["records"])
        self.assertEqual({v["value"]: v["state"] for v in record_views(
            restored["records"], restored["claims"], restored["relations"])},
            {"Chengdu": "historical", "Hangzhou": "active"})


if __name__ == "__main__":
    unittest.main()
