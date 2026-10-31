import json
import unittest

from examples.benchmark_mcp import parse_action
from examples.benchmark_prompt_only import parse_prediction


class ResponseParsingTests(unittest.TestCase):
    def test_explicit_abstain_action_preserves_decision_and_partial_citations(self):
        raw = {"action": "abstain", "answer": "INSUFFICIENT_EVIDENCE [E2][E1]", "abstain": True}
        data = {"question": {"question_id": "q_test"}, "evidence": [{"id": "s_1"}, {"id": "s_2"}]}
        prediction, action, unknown = parse_prediction({"answer": json.dumps(raw)}, data)
        self.assertTrue(prediction["abstain"])
        self.assertEqual(prediction["answer"], "")
        self.assertEqual(set(prediction["citations"]), {"s_1", "s_2"})
        self.assertEqual(action["answer"], raw["answer"])
        self.assertEqual(action["original_action"], "abstain")
        self.assertEqual(unknown, [])

    def test_contradictory_abstain_envelope_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            parse_action(json.dumps({"action": "abstain", "answer": "Miso", "abstain": False}))


if __name__ == "__main__":
    unittest.main()
