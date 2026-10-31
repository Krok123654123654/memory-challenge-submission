"""Local candidate search and relevance filtering. Scores are NOT answerability probabilities.

No Reactor calls, answer generation, or access to gold labels happens here. Keep the full
audit: a rejected candidate is different from proof that the history has no answer.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
STOP = set("a an the i me my you your we our it its is are was were be been do does did "
           "what which who when where how of for to in on at and or with that this please".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


def date(value: str) -> datetime:
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized dataset date: {value!r}")


@dataclass(frozen=True)
class Fragment:
    id: str
    date: str
    turn_id: str
    start: int
    end: int
    text: str
    turn_index: int = 0

    def evidence(self) -> dict:
        return {"id": self.id, "date": self.date, "text": self.text}


def fragments(history: dict, window_words: int = 140, overlap_words: int = 40) -> list[Fragment]:
    """Overlapping windows cover long turns; neighboring context retains its original role."""
    if not 0 <= overlap_words < window_words:
        raise ValueError("overlap_words must be smaller than window_words")
    result = []
    for session in history["sessions"]:
        turns = session["turns"]
        for i, turn in enumerate(turns):
            matches = list(re.finditer(r"\S+", turn["content"]))
            for start in range(0, len(matches), window_words - overlap_words):
                end = min(start + window_words, len(matches))
                body = turn["content"][matches[start].start():matches[end - 1].end()]
                parts = []
                if start == 0 and i:
                    previous = turns[i - 1]
                    parts.append(f"{previous['role']} (preceding context): " +
                                 " ".join(previous["content"].split()[-40:]))
                parts.append(f"{turn['role']}: {body}")
                if end == len(matches) and i + 1 < len(turns):
                    following = turns[i + 1]
                    parts.append(f"{following['role']} (following context): " +
                                 " ".join(following["content"].split()[:20]))
                result.append(Fragment(session["session_id"], session["date"],
                                       turn.get("turn_id", f"turn_{i}"), start, end, "\n".join(parts), i))
                if end == len(matches):
                    break
    return result


class HistoryIndex:
    """BM25 retrieves a broad candidate set; it does not decide evidence relevance."""
    def __init__(self, history: dict):
        self.history_id = history["history_id"]
        self.fragments = fragments(history)
        self.dates = [date(f.date) for f in self.fragments]
        self.lengths = []
        self.postings = defaultdict(list)
        for i, fragment in enumerate(self.fragments):
            counts = Counter(tokens(fragment.text))
            self.lengths.append(sum(counts.values()))
            for word, count in counts.items():
                self.postings[word].append((i, count))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1) or 1

    def search(self, question: dict, count: int = 60) -> list[tuple[Fragment, float]]:
        if question["history_id"] != self.history_id:
            raise ValueError("Question and index belong to different user histories")
        if count < 1:
            raise ValueError("candidate count must be positive")
        scores = defaultdict(float)
        now = date(question["question_date"])
        n = len(self.fragments)
        for word in set(tokens(question["question"])):
            postings = self.postings.get(word, [])
            idf = math.log(1 + (n - len(postings) + .5) / (len(postings) + .5))
            for i, frequency in postings:
                if self.dates[i] <= now:
                    norm = frequency + 1.2 * (.25 + .75 * self.lengths[i] / self.average_length)
                    scores[i] += idf * frequency * 2.2 / norm
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[:count]
        return [(self.fragments[i], scores[i]) for i in ranked]


class LocalReranker:
    def __init__(self, model: str = MODEL, device: str = "cpu", *, revision=None,
                 local_files_only=False, batch_size=16):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install reactor/requirements-retrieval.txt to use local reranking") from exc
        self.name = model
        self.revision = revision or (MODEL_REVISION if model == MODEL else None)
        self.batch_size = batch_size
        self.model = CrossEncoder(model, device=device, max_length=512,
                                  activation_fn=torch.nn.Identity(), trust_remote_code=False,
                                  revision=self.revision,
                                  local_files_only=local_files_only,
                                  model_kwargs={"use_safetensors": True})

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        tokenizer = self.model.tokenizer
        available = 512 - len(tokenizer.encode(question, add_special_tokens=False))
        available -= tokenizer.num_special_tokens_to_add(pair=True) + 8
        if available < 64:
            raise ValueError("Question is too long for this reranker; use a model with a longer context")
        # Score every token window, including long code/URL-heavy passages. Never silently
        # score just the prefix. The best window determines the passage's relevance score.
        pairs, owners = [], []
        for i, passage in enumerate(passages):
            offsets = tokenizer(passage, add_special_tokens=False,
                                return_offsets_mapping=True, verbose=False)["offset_mapping"]
            if not offsets:
                pairs.append((question, passage))
                owners.append(i)
            for start in range(0, len(offsets), available - 32):
                end = min(start + available, len(offsets))
                window = passage[offsets[start][0]:offsets[end - 1][1]]
                if len(tokenizer.encode(question, window)) > 512:
                    raise ValueError("Tokenizer window exceeded its limit; no evidence was selected")
                pairs.append((question, window))
                owners.append(i)
                if end == len(offsets):
                    break
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        if len(scores) != len(pairs) or any(not math.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker returned missing or non-finite window scores")
        result = [float("-inf")] * len(passages)
        for owner, score in zip(owners, scores):
            result[owner] = max(result[owner], float(score))
        return result


def select(question: dict, candidates: list[tuple[Fragment, float]], reranker,
           *, min_score: float, max_excerpts: int = 8, budget: int = 12000) -> dict:
    """Reject weak/duplicate matches. Selected means relevant candidate, not proven support."""
    if not math.isfinite(min_score) or max_excerpts < 1 or budget < 1:
        raise ValueError("Use a finite cutoff and positive excerpt/character limits")
    scores = reranker.score(question["question"], [f.text for f, _ in candidates])
    if len(scores) != len(candidates) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Reranker returned missing or non-finite scores")
    ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
    selected, audit, used = [], [], 0
    for (fragment, retrieval_score), score in ranked:
        cost = len(fragment.text) + len(fragment.id) + len(fragment.date) + 32
        overlaps = any(fragment.id == other.id and fragment.turn_id == other.turn_id and
                       max(0, min(fragment.end, other.end) - max(fragment.start, other.start)) /
                       min(fragment.end - fragment.start, other.end - other.start) > .5
                       for other in selected)
        reason = ("below_cutoff" if score < min_score else
                  "overlapping_fragment" if overlaps else
                  "excerpt_limit" if len(selected) >= max_excerpts else
                  "character_budget" if used + cost > budget else "selected")
        if reason == "selected":
            selected.append(fragment)
            used += cost
        audit.append({**asdict(fragment), "retrieval_score": retrieval_score,
                      "relevance_score": score, "decision": reason})
    selected.sort(key=lambda f: (date(f.date), f.id, f.turn_index, f.start))
    return {"question_id": question["question_id"], "history_id": question["history_id"],
            "question": question["question"], "model": reranker.name,
            "model_revision": getattr(reranker, "revision", None), "min_score": min_score,
            "status": "selected_unverified" if selected else "no_qualifying_evidence",
            "coverage_verified": False, "evidence": [f.evidence() for f in selected],
            "candidates": audit}
