"""No model downloads: verify pruning, session scope and shared packing."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from openprovence_backend import OpenProvenceRetriever
from retrieval import Fragment
from examples.benchmark_rerank_prune import OpenProvence, split_units


class BackendTests(unittest.TestCase):
    def engine(self):
        engine = OpenProvenceRetriever.__new__(OpenProvenceRetriever)
        engine.device = "cpu"
        engine.indexes = {}
        f = Fragment(id="s", date="2024-01-01", turn_id="t", turn_index=0,
                     start=0, end=12, text="user: I found the screwdriver. Unrelated weather discussion.")
        class Embedding:
            def index(self, history): return object()
            def candidates(self, question, history, built, allowed_ids=None):
                self.allowed_ids = allowed_ids
                return [f] if allowed_ids is None or "s" in allowed_ids else []
        class Pruner:
            pruned = staticmethod(OpenProvence.pruned)
            def score(self, query, fragments):
                return [1.0], [split_units(f.text)], [[0.9, 0.01]]
        engine.embedding, engine.pruner = Embedding(), Pruner()
        return engine

    def test_prunes_text_preserves_role_and_source_and_dates(self):
        engine = self.engine()
        q = {"question_id":"q", "history_id":"h", "question":"Did I find it?", "question_date":"2024-01-03"}
        result = engine.search(q, {"history_id":"h"}, allowed_ids={"s"})
        self.assertEqual(engine.embedding.allowed_ids, {"s"})
        self.assertEqual(result["passages"], 1)
        ev = result["evidence"][0]
        self.assertEqual(ev["id"], "s")
        self.assertIn("user: I found the screwdriver.", ev["text"])
        self.assertNotIn("weather", ev["text"])
        self.assertIn("2 calendar days", ev["text"])
        self.assertLessEqual(result["message_chars"], 16000)

    def test_empty_filter_and_wrong_user(self):
        engine = self.engine()
        q = {"question_id":"q", "history_id":"h", "question":"Did I find it?", "question_date":"2024-01-03"}
        self.assertEqual(engine.search(q,{"history_id":"h"},allowed_ids=set())["evidence"], [])
        with self.assertRaises(ValueError):
            engine.search(q,{"history_id":"another-user"})


if __name__ == "__main__":
    unittest.main()
