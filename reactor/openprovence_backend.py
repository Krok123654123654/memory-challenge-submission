"""BGE-large/BM25 retrieval and pinned OpenProvence sentence pruning.

No gold answers, friend predictions or Reactor credentials are read here.
The model adapters are shared with the measured offline experiment.
"""
from pathlib import Path
import time

from evidence_packing import pack_fragments
from model_config import EMBEDDER_PIN, RERANKER_PIN, RETRIEVAL_PROFILE, PRUNING_THRESHOLD


class OpenProvenceRetriever:
    def __init__(self, cache, device="auto"):
        import transformers
        if int(transformers.__version__.split('.')[0]) >= 5:
            raise RuntimeError("OpenProvence needs the compatible environment in requirements-retrieval.txt (Transformers 4.57.6).")
        from examples.benchmark_rerank_prune import Embedder, OpenProvence, pick_device
        self.device = pick_device(device)
        self.embedding = Embedder("bge-large", self.device, Path(cache))
        self.pruner = OpenProvence(self.device)
        if self.embedding.sha != EMBEDDER_PIN["revision"] or self.pruner.sha != RERANKER_PIN["revision"]:
            raise ValueError("Loaded model snapshots differ from the active profile")
        self.indexes = {}

    def search(self, question, history, *, query=None, allowed_ids=None, limit=12, budget=None, audit=False):
        query = question["question"] if query is None else query
        t0 = time.perf_counter()
        hid = history["history_id"]
        if hid != question["history_id"]:
            raise ValueError("Question and history do not belong to the same user")
        if hid not in self.indexes:
            self.indexes[hid] = self.embedding.index(history)
        indexed = time.perf_counter()
        fragments = self.embedding.candidates({**question, "question": query}, history,
                                             self.indexes[hid], allowed_ids=allowed_ids)
        retrieved = time.perf_counter()
        candidate_audit = []
        if fragments:
            scores, parsed, probabilities = self.pruner.score(query, fragments)
            if audit:
                from dataclasses import asdict
                candidate_audit = [{"fragment": asdict(f), "score": float(score),
                                    "units": p[0], "positions": p[1], "labels": p[2],
                                    "sentence_probabilities": [float(x) for x in probs]}
                                   for f, score, p, probs in zip(fragments, scores, parsed, probabilities)]
            order = sorted(range(len(fragments)), key=lambda i: (-scores[i], i))
            ranked = self.pruner.pruned([fragments[i] for i in order], [parsed[i] for i in order],
                                       [probabilities[i] for i in order], PRUNING_THRESHOLD)
        else:
            ranked = []
        pruned = time.perf_counter()
        packed = pack_fragments(question, ranked, limit=limit, evidence_budget=budget)
        return {**packed, **({"candidate_audit": candidate_audit} if audit else {}), "retrieval_profile": RETRIEVAL_PROFILE,
                "embedding": EMBEDDER_PIN["id"], "embedding_revision": EMBEDDER_PIN["revision"],
                "reranker": RERANKER_PIN["id"], "reranker_revision": RERANKER_PIN["revision"],
                "pruning_threshold": PRUNING_THRESHOLD, "device": self.device,
                "candidate_sessions": sorted({f.id for f in fragments}),
                "candidate_fragments": len(fragments), "nonempty_pruned_fragments": len(ranked),
                "index_s": indexed-t0, "candidate_search_s": retrieved-indexed,
                "ranking_pruning_s": pruned-retrieved, "seconds": time.perf_counter()-t0,
                "date_filter_semantics": "inclusive chat dates, not event dates"}
