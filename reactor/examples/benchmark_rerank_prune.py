"""Offline comparison of embedders, rerankers and OpenProvence pruning. No Reactor calls.

For each embedder, every method gets the SAME 40 candidates per question (dense + BM25, reciprocal
rank fusion, top 40 -- the same recipe as search_backend.SearchBackend.search_history). Only the
ranking / pruning / packing step differs. Gold is used only to score results, never to select.

Embedders (candidate generation)
  bge-large  BAAI/bge-large-en-v1.5 (MIT; selected for this GPU comparison)
  arctic     Snowflake/snowflake-arctic-embed-s (Apache-2.0; optional historical baseline only)

Methods (after candidates)
  bge-base-12           current: BGE reranker base, original text, max 12 passages
  bge-large-12          BGE reranker large (MIT), original text, max 12 passages
  op-rank-12            OpenProvence reranking score, original text, max 12 passages
  op-prune-fill@T       OpenProvence rank + prune at threshold T, no passage cap, fill 16k message
  bge-large+op-prune@T  BGE reranker large order, OpenProvence-pruned text, no passage cap, fill 16k

Run from reactor/:
  python examples/benchmark_rerank_prune.py --device cuda --embedders bge-large
Methods whose model cannot be loaded are skipped with a message; the rest still run.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parent
sys.path.insert(0, str(KIT))

from client import MAX_CHARS  # noqa: E402
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION  # noqa: E402
from evidence_packing import annotated, answer_message  # noqa: E402
from retrieval import Fragment, HistoryIndex, LocalReranker, date  # noqa: E402

EMBEDDERS = {
    "arctic": ("Snowflake/snowflake-arctic-embed-s", "e596f507467533e48a2e17c007f0e1dacc837b33",
               "Represent this sentence for searching relevant passages: "),
    "bge-large": ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                  "Represent this sentence for searching relevant passages: "),
}
BASE_ID, BASE_REV = "BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"
LARGE_ID = "BAAI/bge-reranker-large"
LARGE_REV = "55611d7bca2a7133960a6d3b71e083071bbfc312"
OP_ID = "hotchpotch/open-provence-reranker-v1-gte-modernbert-base"
OP_REVIEWED_SHA = "911517e2dc847a8ec00ceae795cd89e86ef90319"
THRESHOLDS = (0.05, 0.1, 0.2)
ROLE = re.compile(r"(?m)^(?=(?:user|assistant)(?: \((?:preceding|following) context\))?: )")
LABEL = re.compile(r"^((?:user|assistant)(?: \((?:preceding|following) context\))?: )")
SENT = re.compile(r"(?<=[.!?])\s+(?=\S)")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:  # physical lines; U+2028 may occur in strings
        return [json.loads(line) for line in f if line.strip()]


def snapshot(repo, revision=None):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision, allow_patterns=[
        "*.json", "*.safetensors", "*.txt", "*.model", "*.py", "README.md", "LICENSE*",
    ]))


def pick_device(name):
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- data
def prepare(out: Path, sample: Path, dev: Path):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "questions.jsonl").exists():
        return
    questions = json.loads(sample.read_text())
    wanted = {q["history_id"] for q in questions}
    with (out / "questions.jsonl").open("w") as f:
        for q in questions:
            f.write(json.dumps({k: q[k] for k in ("question_id", "history_id", "question",
                                                  "question_date")}) + "\n")
    found = 0
    with (dev / "histories.jsonl").open(encoding="utf-8-sig") as src, \
            (out / "histories.jsonl").open("w") as dst:
        for line in src:
            if line.strip():
                h = json.loads(line)
                if h["history_id"] in wanted:
                    dst.write(json.dumps(h) + "\n")
                    found += 1
    if found != len(wanted):
        raise SystemExit(f"Found {found} of {len(wanted)} histories in {dev}/histories.jsonl")


def allowed(q, history):
    now = date(q["question_date"])
    return {s["session_id"] for s in history["sessions"] if date(s["date"]) <= now}


# ---------------------------------------------------------------- candidates
def windows(tokenizer, text, max_tokens=480):
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        verbose=False)["offset_mapping"]
    if not offsets:
        return [text]
    result = []
    for start in range(0, len(offsets), max_tokens - 48):
        end = min(start + max_tokens, len(offsets))
        result.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == len(offsets):
            break
    return result


class Embedder:
    def __init__(self, key, device, cache: Path):
        import torch
        from sentence_transformers import SentenceTransformer
        repo, rev, self.prompt = EMBEDDERS[key]
        path = snapshot(repo, rev)
        self.key, self.sha, self.cache = key, path.name, cache
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = SentenceTransformer(str(path), device=device, trust_remote_code=False,
                                         model_kwargs={"dtype": dtype, "use_safetensors": True})
        self.model.max_seq_length = 512
        self.batch = 128 if device == "cuda" else 16

    def index(self, history):
        import numpy as np
        idx = HistoryIndex(history)
        texts = [f.text for f in idx.fragments]
        fp = hashlib.sha256(json.dumps([self.key, self.sha, 480, 48, texts]).encode()).hexdigest()
        path = self.cache / f"{self.key}-{fp[:24]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as d:
                return idx, d["vectors"], d["owners"]
        pieces, owners = [], []
        for i, text in enumerate(texts):
            parts = windows(self.model.tokenizer, text)
            pieces.extend(parts)
            owners.extend([i] * len(parts))
        vectors = self.model.encode(pieces, batch_size=self.batch, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True).astype("float32")
        owners = np.array(owners, dtype="int64")
        if not np.isfinite(vectors).all():
            raise ValueError("Non-finite embedding")
        self.cache.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=vectors, owners=owners)
        return idx, vectors, owners

    def candidates(self, q, history, built, allowed_ids=None):
        import numpy as np
        ok = allowed(q, history)
        if allowed_ids is not None:
            ok &= set(allowed_ids)
        idx, vectors, owners = built
        valid = [i for i, f in enumerate(idx.fragments) if f.id in ok]
        query_fp = hashlib.sha256(json.dumps([self.key, self.sha, self.prompt, 512, q["question"]]).encode()).hexdigest()
        query_path = self.cache / f"query-{query_fp[:24]}.npy"
        if query_path.exists():
            vec = np.load(query_path, allow_pickle=False)
        else:
            vec = self.model.encode([q["question"]], prompt=self.prompt, normalize_embeddings=True,
                                    show_progress_bar=False)[0].astype("float32")
            self.cache.mkdir(parents=True, exist_ok=True)
            np.save(query_path, vec)
        if vec.shape != (vectors.shape[1],) or not np.isfinite(vec).all():
            raise ValueError("Invalid cached query embedding")
        dense_scores = np.full(len(idx.fragments), -np.inf, dtype="float32")
        np.maximum.at(dense_scores, owners, vectors @ vec)
        dense = sorted(valid, key=lambda i: (-dense_scores[i], i))[:60]
        positions = {(f.id, f.turn_id, f.start, f.end): i for i, f in enumerate(idx.fragments)}
        lexical = idx.search(q, len(idx.fragments))
        bm = [positions[(f.id, f.turn_id, f.start, f.end)] for f, _ in lexical if f.id in ok][:60]
        fusion = defaultdict(float)
        for ordering in (dense, bm):
            for rank, i in enumerate(ordering, 1):
                fusion[i] += 1 / (60 + rank)
        return [idx.fragments[i] for i in sorted(fusion, key=lambda i: (-fusion[i], i))[:40]]


# ---------------------------------------------------------------- packing
def pack(q: dict, ranked: list[Fragment], max_passages: int | None):
    """Same rules as evidence_packing.pack_fragments, but the passage cap is optional."""
    selected, evidence, skipped = [], [], 0
    for f in ranked:
        if max_passages is not None and len(selected) == max_passages:
            break
        overlap = any(f.id == o.id and f.turn_id == o.turn_id and
                      max(0, min(f.end, o.end) - max(f.start, o.start)) /
                      max(1, min(f.end - f.start, o.end - o.start)) > .5 for o in selected)
        if overlap or not f.text.strip():
            continue
        ordered = sorted(selected + [f], key=lambda x: (date(x.date), x.id, x.turn_index, x.start))
        proposed = [annotated(q, x) for x in ordered]
        if len(answer_message(q, proposed, "00000000")) > MAX_CHARS:
            skipped += 1
            continue
        selected.append(f)
        evidence = proposed
    return evidence, skipped


# ---------------------------------------------------------------- OpenProvence
def split_units(text: str):
    """Role segments -> lines -> sentences. Returns units and their (segment, line) positions."""
    segments = [s for s in ROLE.split(text) if s.strip()]
    units, where, labels = [], [], []
    for si, seg in enumerate(segments):
        m = LABEL.match(seg)
        labels.append(m.group(1) if m else "")
        body = seg[m.end():] if m else seg
        for li, line in enumerate(body.split("\n")):
            for sent in SENT.split(line.strip()):
                if sent.strip():
                    units.append(sent.strip())
                    where.append((si, li))
    return units, where, labels


def rebuild(units, where, labels, keep):
    """Kept sentences only; speaker labels kept; '[...]' marks removed text."""
    segs = defaultdict(lambda: defaultdict(list))
    last = None
    for i, (u, w, k) in enumerate(zip(units, where, keep)):
        if not k:
            continue
        if last is not None and i != last + 1:
            segs[w[0]][w[1]].append("[...]")
        segs[w[0]][w[1]].append(u)
        last = i
    out = []
    for si in sorted(segs):
        lines = [" ".join(segs[si][li]) for li in sorted(segs[si])]
        out.append(labels[si] + "\n".join(lines))
    return "\n".join(out)


class OpenProvence:
    def __init__(self, device):
        from transformers import AutoModel
        path = snapshot(OP_ID, OP_REVIEWED_SHA)
        code = (path / "modeling_open_provence_standalone.py").read_bytes()
        self.sha = path.name
        self.code_sha256 = hashlib.sha256(code).hexdigest()
        if self.sha != OP_REVIEWED_SHA:
            raise ValueError("OpenProvence must use the reviewed snapshot")
        # trust_remote_code: the checkpoint ships its own model class (reviewed: no network/exec).
        # Transformers 4 checks the wrapper's declared attention support. The
        # custom wrapper does not advertise SDPA; use its supported loader path.
        kw = {"attn_implementation": "eager"} if device == "cuda" else {}
        self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True, device=device, **kw)

    def score(self, question: str, frags: list[Fragment]):
        parsed = [split_units(f.text) for f in frags]
        res = self.model.process(question=question, context=[p[0] or [f.text] for p, f in zip(parsed, frags)],
                                 title=None, batch_size=32, threshold=0.1, show_progress=False,
                                 return_sentence_metrics=True, zero_score_when_empty=False,
                                 enable_warnings=False)
        scores = [float(s) if s is not None else float("-inf") for s in res["reranking_score"]]
        probs = res["sentence_probabilities"]
        if len(scores) != len(frags) or len(probs) != len(frags):
            raise ValueError("OpenProvence returned an incomplete candidate batch")
        for (units, _, _), p in zip(parsed, probs):
            if units and len(p) != len(units):
                raise ValueError(f"OpenProvence returned {len(p)} probabilities for {len(units)} sentences")
        return scores, parsed, probs

    @staticmethod
    def pruned(frags, parsed, probs, threshold):
        result = []
        for f, (units, where, labels), p in zip(frags, parsed, probs):
            text = rebuild(units, where, labels, [x > threshold for x in p]) if units else ""
            if text.strip():
                result.append(dataclasses.replace(f, text=text))
        return result


# ---------------------------------------------------------------- scoring
def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/rerank-prune-20261031")
    ap.add_argument("--sample", default="benchmarks/as-is-20261031/questions.json")
    ap.add_argument("--dev", default=str(ROOT / "dev"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--embedders", default="bge-large", help="comma list; bge-large is the selected embedder")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    args = ap.parse_args()
    out = (KIT / args.out).resolve()
    dev = Path(args.dev).resolve()
    prepare(out, (KIT / args.sample).resolve(), dev)
    device = pick_device(args.device)
    print(f"device={device}", flush=True)
    questions = read_jsonl(out / "questions.jsonl")[:args.limit]
    if not questions:
        raise SystemExit("No questions selected")
    # Append each completed measurement immediately, even if a later model fails.
    checkpoint = out / "measurements.partial.jsonl"
    checkpoint.write_text("")
    histories = {h["history_id"]: h for h in read_jsonl(out / "histories.jsonl")}
    gold = {g["question_id"]: g for g in read_jsonl(dev / "gold.jsonl")}

    # ---- candidates per embedder
    cands, shas = {}, {}
    for key in args.embedders.split(","):
        emb = Embedder(key, device, out / "cache")
        shas[key] = emb.sha
        t0 = time.perf_counter()
        built = {}
        cands[key] = {}
        for n, q in enumerate(questions, 1):
            h = histories[q["history_id"]]
            if q["history_id"] not in built:
                built[q["history_id"]] = emb.index(h)
            cands[key][q["question_id"]] = emb.candidates(q, h, built[q["history_id"]])
            if n % 10 == 0 or n == len(questions):
                print(f"[{key}] candidates {n}/{len(questions)}  {time.perf_counter()-t0:.0f}s", flush=True)
        del emb, built
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    # ---- rerankers (skip any that fail to load)
    models, errors = {}, []
    for name, repo, rev in (("bge-base", BASE_ID, BASE_REV), ("bge-large", LARGE_ID, LARGE_REV)):
        try:
            models[name] = LocalReranker(str(snapshot(repo, rev)), device=device,
                                         batch_size=32 if device == "cuda" else 4)
        except Exception as e:  # noqa: BLE001
            errors.append({"model": name, "error": f"{type(e).__name__}: {e}"})
            print(f"SKIP {name}: {type(e).__name__}: {e}")
    op = None
    try:
        op = OpenProvence(device)
    except Exception as e:  # noqa: BLE001
        errors.append({"model": "OpenProvence", "error": f"{type(e).__name__}: {e}"})
        print(f"SKIP OpenProvence: {type(e).__name__}: {e}")
    (out / "model-errors.json").write_text(json.dumps(errors, indent=2))

    rows = defaultdict(list)
    cand_recall = defaultdict(list)
    for key in cands:
        for n, q in enumerate(questions, 1):
            qid, fs = q["question_id"], cands[key][q["question_id"]]
            g = gold[qid]
            hist = histories[q["history_id"]]
            gs = set(g["evidence"])
            if not g["abstain"]:
                cand_recall[key].append(len({f.id for f in fs} & gs) / len(gs))
            full = {s["session_id"]: norm(" ".join(t["content"] for t in s["turns"])) for s in hist["sessions"]}
            answers = [norm(a) for a in [g["answer"], *g.get("aliases", [])] if 0 < len(norm(a)) <= 40]
            checkable = [a for a in answers if any(a in full.get(s, "") for s in gs)]

            def record(method, ranked, cap, seconds, top):
                ev, skipped = pack(q, ranked, cap)
                sent = {e["id"] for e in ev}
                blob = norm(" ".join(e["text"] for e in ev))
                r = {"question_id": qid, "type": g["question_type"], "abstain_gold": g["abstain"],
                     "passages": len(ev), "message_chars": len(answer_message(q, ev, "00000000")),
                     "skipped_for_budget": skipped, "seconds": seconds, "top_score": top,
                     "sent_sessions": sorted(sent), "evidence": ev}
                if not g["abstain"]:
                    tp = len(sent & gs)
                    p = tp / len(sent) if sent else 0.0
                    rc = tp / len(gs) if gs else 0.0
                    r.update(precision=p, recall=rc, f1=2 * p * rc / (p + rc) if p + rc else 0.0,
                             all_gold=gs <= sent,
                             answer_text_sent=(any(a in blob for a in checkable) if checkable else None))
                rows[f"{key}/{method}"].append(r)
                with checkpoint.open("a") as f:
                    f.write(json.dumps({"method": f"{key}/{method}", **r}) + "\n")

            def order(scores):
                return sorted(range(len(fs)), key=lambda i: (-scores[i], i))

            large = None
            for name, model in models.items():
                t = time.perf_counter()
                s = model.score(q["question"], [f.text for f in fs])
                dt = time.perf_counter() - t
                record(f"{name}-12", [fs[i] for i in order(s)], 12, dt, max(s))
                if name == "bge-large":
                    large = (s, dt)
            if op is not None:
                t = time.perf_counter()
                s, parsed, probs = op.score(q["question"], fs)
                dt = time.perf_counter() - t
                record("op-rank-12", [fs[i] for i in order(s)], 12, dt, max(s))
                o = order(s)
                for th in THRESHOLDS:
                    pr = OpenProvence.pruned([fs[i] for i in o], [parsed[i] for i in o],
                                             [probs[i] for i in o], th)
                    record(f"op-prune-fill@{th}", pr, None, dt, max(s))
                    record(f"op-prune-12@{th}", pr, 12, dt, max(s))
                if large is not None:
                    o = order(large[0])
                    for th in (0.05, 0.1):
                        pr = OpenProvence.pruned([fs[i] for i in o], [parsed[i] for i in o],
                                                 [probs[i] for i in o], th)
                        record(f"bge-large+op-prune@{th}", pr, None, large[1] + dt, max(large[0]))
                        record(f"bge-large+op-prune-12@{th}", pr, 12, large[1] + dt, max(large[0]))
            if n % 10 == 0 or n == len(questions):
                print(f"[{key}] ranked {n}/{len(questions)}", flush=True)

    # ---- summary
    summary = {"questions": len(questions), "device": device, "candidate_pool": 40,
               "status": "partial" if errors else "complete", "model_errors": errors,
               "prompt_version": ANSWER_PROMPT_VERSION, "guidance_chars": len(ANSWER_GUIDANCE),
               "guidance_sha256": hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest(),
               "message_limit": MAX_CHARS, "embedder_snapshots": shas,
               "reranker_snapshots": {"bge-base": BASE_REV, "bge-large": LARGE_REV},
               "openprovence_snapshot": getattr(op, "sha", None),
               "openprovence_code_sha256": getattr(op, "code_sha256", None),
               "candidate_session_recall": {k: statistics.mean(v) for k, v in cand_recall.items()},
               "note": "Session metrics and answer_text_sent are retrieval proxies, not Reactor answer "
                       "accuracy. answer_text_sent is a literal-match heuristic on short gold answers only. "
                       "Seconds exclude model loading and indexing. Methods with '-12' respect the "
                       "production passage cap; other pruning arms are uncapped experiments.",
               "methods": {}}
    print("\ncandidate session recall (gold sessions among the 40 candidates):",
          {k: round(v, 3) for k, v in summary["candidate_session_recall"].items()})
    print(f"\n{'method':34} {'recall':>6} {'prec':>5} {'F1':>5} {'allgold':>7} {'ans-sent':>9} "
          f"{'pass':>5} {'chars':>6} {'s/q':>5} {'abstAUC':>7}")
    for method, rs in rows.items():
        a = [r for r in rs if not r["abstain_gold"]]
        chk = [r["answer_text_sent"] for r in a if r["answer_text_sent"] is not None]
        m = {"questions": len(rs), "answerable_questions": len(a),
             "recall": statistics.mean(r["recall"] for r in a),
             "precision": statistics.mean(r["precision"] for r in a),
             "f1": statistics.mean(r["f1"] for r in a),
             "all_gold": sum(r["all_gold"] for r in a) / len(a),
             "answer_text_sent": (sum(chk) / len(chk)) if chk else None, "answer_checkable": len(chk),
             "passages": statistics.mean(r["passages"] for r in rs),
             "message_chars": statistics.mean(r["message_chars"] for r in rs),
             "median_seconds": statistics.median(r["seconds"] for r in rs),
             "abstain_auc_top_score": auc([r["top_score"] for r in a],
                                          [r["top_score"] for r in rs if r["abstain_gold"]])}
        summary["methods"][method] = m
        with (out / f"results-{method.replace('/', '__')}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        ans = f"{m['answer_text_sent']:.3f}/{m['answer_checkable']}" if chk else "n/a"
        auc_s = f"{m['abstain_auc_top_score']:.3f}" if m["abstain_auc_top_score"] is not None else "n/a"
        print(f"{method:34} {m['recall']:6.3f} {m['precision']:5.3f} {m['f1']:5.3f} {m['all_gold']:7.3f} "
              f"{ans:>9} {m['passages']:5.1f} {m['message_chars']:6.0f} {m['median_seconds']:5.2f} {auc_s:>7}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nSaved {out / 'summary.json'}")
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
