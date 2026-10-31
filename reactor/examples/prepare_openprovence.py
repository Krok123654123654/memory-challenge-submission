"""Prepare fixed evidence plans with no answers or gold labels as inputs."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from openprovence_backend import OpenProvenceRetriever
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION
from model_config import RETRIEVAL_PROFILE


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-audit", action="store_true", help="retain original candidates and pruning probabilities")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    questions = rows(args.input / "questions.jsonl")
    histories = {h["history_id"]: h for h in rows(args.input / "histories.jsonl")}
    start = time.perf_counter()
    engine = OpenProvenceRetriever(args.out / "cache", device=args.device)
    loaded = time.perf_counter()
    with (args.out / "plans.jsonl").open("w") as f:
        for n, q in enumerate(questions, 1):
            plan = {"question_id": q["question_id"], "history_id": q["history_id"],
                    "question_sha256": hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest(),
                    "prompt_version": ANSWER_PROMPT_VERSION,
                    "guidance_sha256": hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest(),
                    **engine.search(q, histories[q["history_id"]], audit=args.save_audit)}
            if args.save_audit:
                audit_dir = args.out / "audit"
                audit_dir.mkdir(exist_ok=True)
                (audit_dir / f"{q['question_id']}.json").write_text(json.dumps(plan.pop("candidate_audit"), ensure_ascii=False))
            f.write(json.dumps(plan, ensure_ascii=False)+"\n"); f.flush()
            print(json.dumps({"prepared": n, "total": len(questions), "question_id": q["question_id"],
                              "passages": plan["passages"], "chars": plan["message_chars"],
                              "index_s": round(plan["index_s"],2),
                              "rank_prune_s": round(plan["ranking_pruning_s"],2)}), flush=True)
    manifest = {"status": "complete", "questions": len(questions), "histories": len(histories),
                "retrieval_profile": RETRIEVAL_PROFILE, "prompt_version": ANSWER_PROMPT_VERSION,
                "guidance_sha256": hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest(),
                "model_load_s": loaded-start, "total_s": time.perf_counter()-start}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
