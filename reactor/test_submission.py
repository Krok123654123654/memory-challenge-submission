import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from answer_guidance import ANSWER_GUIDANCE
from evidence_packing import answer_message
from model_config import RETRIEVAL_PROFILE
from submission import validate_complete, enable_concise


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.qs = [{"question_id": f"q_{i}", "history_id": "h_1", "question": "What did I say?",
                    "question_date": "2023/06/01 (Thu) 12:00"} for i in range(2)]
        self.rows = [{"question_id": q["question_id"], "answer": "hello", "abstain": False, "citations": ["s_1"]} for q in self.qs]
        self.save("protocol.json", {"cases": [q["question_id"] for q in self.qs],
            "retrieval_profile": RETRIEVAL_PROFILE, "guidance_sha256": hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest()})
        self.jsonl("questions.jsonl", self.qs)
        self.jsonl("histories.jsonl", [{"history_id": "h_1", "sessions": [{"session_id": "s_1", "date": "2023/05/31 (Wed) 12:00", "turns": []}]}])
        self.jsonl("predictions.jsonl", self.rows)
        for q, pred in zip(self.qs, self.rows):
            folder = self.root / q["question_id"]
            folder.mkdir()
            evidence = [{"id": "s_1", "date": "2023/05/31 (Wed) 12:00", "text": "user: hello"}]
            self.save(f"{q['question_id']}/input.json", {"question": q, "evidence": evidence, "ref": "12345678",
                "prompt": answer_message(q, evidence, "12345678")})
            self.save(f"{q['question_id']}/result.json", {"prediction": pred, "unknown_citations": [], "history_id": "remote_1"})
            self.save(f"{q['question_id']}/response.json", {"answer": json.dumps({"action": "answer", "answer": "hello [E1]", "abstain": False})})
            self.save(f"{q['question_id']}/state.json", {"pending": False})

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def jsonl(self, name, values):
        (self.root / name).write_text(''.join(json.dumps(x)+'\n' for x in values))

    def test_complete_provenanced_predictions_pass(self):
        self.assertTrue(validate_complete(self.root)["valid"])

    def test_concise_changes_only_unsent_inputs_and_keeps_evidence(self):
        first = (self.root / "q_0/input.json").read_bytes()
        path = self.root / "q_1/input.json"
        original = json.loads(path.read_text())
        for name in ("result.json", "response.json", "state.json"):
            (self.root / "q_1" / name).unlink()
        self.assertEqual(enable_concise(self.root)["concise_unsent_questions"], 1)
        changed = json.loads(path.read_text())
        self.assertEqual(first, (self.root / "q_0/input.json").read_bytes())
        self.assertEqual(changed["evidence"], original["evidence"])
        self.assertEqual(changed["ref"], original["ref"])
        self.assertLessEqual(len(changed["prompt"]), len(original["prompt"]))
        self.assertEqual(changed["response_style"], "concise-v1")

    def test_concise_never_changes_pending_request(self):
        for name in ("result.json", "response.json"):
            (self.root / "q_1" / name).unlink()
        self.save("q_1/state.json", {"pending": True})
        before = (self.root / "q_1/input.json").read_bytes()
        self.assertEqual(enable_concise(self.root)["concise_unsent_questions"], 0)
        self.assertEqual(before, (self.root / "q_1/input.json").read_bytes())

    def test_missing_prediction_is_not_converted_to_abstention(self):
        self.jsonl("predictions.jsonl", self.rows[:1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_complete(self.root)

    def test_answer_edits_are_rejected(self):
        self.rows[1]["answer"] = "edited answer"
        self.jsonl("predictions.jsonl", self.rows)
        with self.assertRaisesRegex(AssertionError, "changed after parsing"):
            validate_complete(self.root)

    def test_mid_user_reset_is_rejected(self):
        self.save("q_1/result.json", {"prediction": self.rows[1], "unknown_citations": [], "history_id": "different_remote"})
        with self.assertRaisesRegex(AssertionError, "reset mid-run"):
            validate_complete(self.root)

    def test_foreign_citation_is_rejected_even_if_present_in_raw_result(self):
        self.rows[1]["citations"] = ["s_other_user"]
        self.jsonl("predictions.jsonl", self.rows)
        self.save("q_1/result.json", {"prediction": self.rows[1], "unknown_citations": [], "history_id": "remote_1"})
        with self.assertRaises(AssertionError):
            validate_complete(self.root)


if __name__ == "__main__":
    unittest.main()
