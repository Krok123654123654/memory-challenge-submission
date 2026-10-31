import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from examples import benchmark_long_multiturn as benchmark
from examples import benchmark_prompt_only as transport


class SharedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        credentials = self.root / ".config/reactor/credentials.json"
        credentials.parent.mkdir(parents=True)
        credentials.write_text(json.dumps({"base_url": "https://invalid.example", "agent_id": "fake", "access_key": "fake-key"}))
        credentials.chmod(0o600)
        self.qs = [{"question_id": f"q_{i}", "history_id": hid} for i, hid in enumerate(("h1", "h1", "h2"))]
        for q in self.qs:
            folder = self.root / q["question_id"]
            folder.mkdir()
            (folder / "input.json").write_text(json.dumps({"question": q, "ref": q["question_id"],
                "prompt": "answer the question", "evidence": [], "shown": []}))
        self.home = patch.object(transport.Path, "home", return_value=self.root)
        self.home.start()
        self.addCleanup(self.home.stop)
        self.report = patch.object(benchmark, "report")
        self.report.start()
        self.addCleanup(self.report.stop)

    @staticmethod
    def response():
        return {"answer": json.dumps({"action": "answer", "answer": "ok", "abstain": False}), "late_replies": []}

    def test_one_remote_history_per_user_and_completed_resume_is_noop(self):
        with patch.object(transport.client, "new_history", side_effect=[{"history_id": "r1"}, {"history_id": "r2"}]) as new, \
             patch.object(transport.client, "ask_retrying", side_effect=lambda *a, **k: (self.response(), 0)) as ask:
            self.assertEqual(benchmark.execute(self.root, self.qs), 0)
            self.assertEqual(new.call_count, 2)
            self.assertEqual([call.kwargs["history"] for call in ask.call_args_list], ["r1", "r1", "r2"])
            self.assertEqual(benchmark.execute(self.root, self.qs), 0)
            self.assertEqual(ask.call_count, 3)
            self.assertEqual(new.call_count, 2)

    def test_timeout_is_received_in_same_history_before_next_question(self):
        with patch.object(transport.client, "new_history", return_value={"history_id": "r1"}) as new, \
             patch.object(transport.client, "ask_retrying", side_effect=TimeoutError("uncertain delivery")) as first:
            self.assertEqual(benchmark.execute(self.root, self.qs[:2]), 3)
            self.assertEqual(first.call_count, 1)
            self.assertEqual(new.call_count, 1)
        with patch.object(transport.client, "new_history") as reset, \
             patch.object(transport.client, "ask", new_callable=AsyncMock, return_value=self.response()) as receive, \
             patch.object(transport.client, "ask_retrying", return_value=(self.response(), 0)) as next_question:
            self.assertEqual(benchmark.execute(self.root, self.qs[:2]), 0)
            reset.assert_not_called()
            receive.assert_awaited_once()
            self.assertTrue(receive.call_args.kwargs["receive_only"])
            self.assertEqual(receive.call_args.kwargs["history"], "r1")
            self.assertEqual(receive.call_args.kwargs["ref"], "q_0")
            self.assertEqual(next_question.call_count, 1)
            self.assertEqual(next_question.call_args.kwargs["history"], "r1")

    def test_pending_request_blocks_another_users_history(self):
        (self.root / "histories-state.json").write_text(json.dumps({"histories": {"h1": "r1"}, "active_dataset_history": "h1"}))
        (self.root / "q_0/state.json").write_text(json.dumps({"history_id": "r1", "pending": True}))
        with patch.object(transport, "run") as send:
            with self.assertRaisesRegex(ValueError, "Unresolved request"):
                benchmark.execute(self.root, self.qs[2:])
            send.assert_not_called()

    def test_replaced_history_cannot_be_recreated(self):
        (self.root / "histories-state.json").write_text(json.dumps({"histories": {"h1": "r1", "h2": "r2"}, "active_dataset_history": "h2"}))
        with patch.object(transport, "run") as send:
            with self.assertRaisesRegex(ValueError, "replaced Reactor history"):
                benchmark.execute(self.root, self.qs[:1])
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
