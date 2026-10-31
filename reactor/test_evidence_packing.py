"""Check full-message boundaries, source identity and candidate packing."""
import unittest
from evidence_packing import annotated, answer_message, pack_fragments
from retrieval import Fragment


class PackingTests(unittest.TestCase):
    q = {"question": "What changed?", "question_date": "2024-03-01"}

    def f(self, i, text="Evidence", **kw):
        return Fragment(**({"id": f"s_{i}", "date": "2024-02-29", "turn_id": "t_0",
                            "start": 0, "end": 100, "text": text} | kw))

    def test_exact_message_boundary(self):
        f = self.f(0)
        size = len(answer_message(self.q, [annotated(self.q, f)], "abcdefgh"))
        packed = pack_fragments(self.q, [f], max_chars=size)
        self.assertEqual(packed["passages"], 1)
        self.assertEqual(packed["remaining_chars"], 0)
        self.assertEqual(pack_fragments(self.q, [f], max_chars=size-1)["passages"], 0)

    def test_cap_and_multidigit_labels(self):
        packed = pack_fragments(self.q, [self.f(i) for i in range(20)])
        prompt = answer_message(self.q, packed["evidence"], "abcdefgh")
        self.assertEqual(packed["passages"], 12)
        self.assertEqual(packed["message_chars"], len(prompt))
        self.assertIn("[E12]", prompt)

    def test_oversize_skipped_not_truncated(self):
        packed = pack_fragments(self.q, [self.f(0, "x"*16000), self.f(1, "Small fact")])
        self.assertEqual([e["id"] for e in packed["evidence"]], ["s_1"])
        self.assertEqual(packed["skipped_for_budget"], 1)
        self.assertTrue(packed["evidence"][0]["text"].endswith("Small fact"))

    def test_overlap_and_one_per_session(self):
        fs = [self.f(0), self.f(0, start=20), self.f(0, turn_id="t_1")]
        self.assertEqual(pack_fragments(self.q, fs)["passages"], 2)
        self.assertEqual(pack_fragments(self.q, fs, one_per_session=True)["passages"], 1)

    def test_chronological_and_optional_evidence_budget(self):
        fs = [self.f(0), self.f(1, date="2024-01-01")]
        p = pack_fragments(self.q, fs)
        self.assertEqual([e["id"] for e in p["evidence"]], ["s_1", "s_0"])
        self.assertEqual(pack_fragments(self.q, fs, evidence_budget=0)["passages"], 0)
        with self.assertRaises(ValueError):
            pack_fragments(self.q, fs, limit=13)


if __name__ == "__main__": unittest.main()
