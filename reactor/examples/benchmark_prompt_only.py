"""Repeat the two completed fixed-retrieval cases with only task guidance changed.

Preparation copies the original wire prompts and replaces their role/goal paragraph.
Evidence, ordering, dates, output instructions and prediction parsing stay fixed.
Run only after the team confirms the agent is free. Saved requests resume receive-only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parent))
import client
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION
from examples.benchmark_mcp import parse_action, save

CASES = ("q_579e6012dcfb", "q_466e22735f43")
BASELINE = KIT / "benchmarks/mcp-paired-20261031"
DEFAULT = KIT / "benchmarks/prompt-only-v5-20261031"


def parse_prediction(response, data):
    action = parse_action(response["answer"])
    assert action["action"] == "answer", "This experiment has no tools"
    answer = action["answer"]
    cited, unknown = client.citations(answer, data["evidence"])
    abstain = action["abstain"] or bool(client._INSUFFICIENT.match(answer))
    cleaned = re.sub(r"\[(?:E\d+[\s,]*)+\]", "", answer).strip()
    prediction = {"question_id": data["question"]["question_id"], "answer": "" if abstain else cleaned,
                  "abstain": abstain, "citations": list(dict.fromkeys(c["id"] for c in cited))}
    return prediction, action, unknown


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []
    for qid in CASES:
        folder = args.out / qid
        folder.mkdir()
        original = json.loads((BASELINE / f"{qid}-fixed/state.json").read_text())
        result = json.loads((BASELINE / f"{qid}-fixed/result.json").read_text())
        old = original["steps"][0]
        prefix, suffix = old["prompt"].split("\nEvidence:\n", 1)
        header = "\n".join(prefix.splitlines()[:2])
        ref = secrets.token_hex(4)
        prompt = (header + "\n" + ANSWER_GUIDANCE + "\n\nEvidence:\n" + suffix).replace(
            f"[ref:{old['ref']}]", f"[ref:{ref}]")
        restored_suffix = prompt.split("\nEvidence:\n", 1)[1].replace(
            f"[ref:{ref}]", f"[ref:{old['ref']}]")
        assert restored_suffix == suffix, "Evidence and output contract must remain byte-identical"
        assert len(prompt) <= client.MAX_CHARS
        assert original["shown"] == list(range(8)) and len(original["evidence"]) == 8
        save(folder / "input.json", {"question": result["question"], "ref": ref, "prompt": prompt,
             "evidence": original["evidence"], "shown": original["shown"]})
        save(folder / "baseline.json", result)
        checks.append({"question_id": qid, "prompt_chars_before": len(old["prompt"]),
            "prompt_chars_after": len(prompt), "evidence_and_output_contract_identical": True,
            "original_suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest()})
    save(args.out / "protocol.json", {"prompt_version": ANSWER_PROMPT_VERSION,
        "cases": list(CASES), "changed": "Role, goal and evaluation-context paragraph only; fresh request refs",
        "held_fixed": ["question", "question date", "eight evidence passages, byte-for-byte",
            "evidence order and labels", "date annotations", "JSON output instructions",
            "final-answer-only citation extraction", "600-second timeout", "same team agent",
            "fresh history per distinct dataset history", "sequential execution", "no MCP calls"],
        "guidance": ANSWER_GUIDANCE, "checks": checks,
        "limitations": ["Two selected dev cases, one repeat each; not an unbiased or statistically reliable gain",
                        "Server model settings are not exposed or controlled by this client"]})
    print(json.dumps({"prepared": str(args.out), "cases": checks}), flush=True)


def run(args):
    if not args.agent_free:
        raise SystemExit("Confirm team availability before running with --agent-free.")
    import fcntl
    os.umask(0o077)
    with (args.out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        credentials = Path.home() / ".config/reactor/credentials.json"
        assert credentials.stat().st_mode & 0o077 == 0
        bundle = json.loads(credentials.read_text())
        key = bundle["access_key"]
        for qid in CASES:
            if (args.out / "STOP").exists():
                print("Stopped between requests.", flush=True)
                return
            folder = args.out / qid
            if (folder / "result.json").exists():
                continue
            data = json.loads((folder / "input.json").read_text())
            path = folder / "state.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            if "history_id" not in state:
                if state.get("creating_history"):
                    raise RuntimeError("Interrupted history creation; recover its ID before continuing.")
                state["creating_history"] = True
                save(path, state)
                started = time.monotonic()
                history = client.new_history(bundle["base_url"], bundle["agent_id"], key, timeout=600)
                state.update(history_id=history["history_id"], creating_history=False,
                             history_setup_s=time.monotonic() - started)
                save(path, state)
            receive_only = state.get("pending", False) and not state.get("not_forwarded", False)
            cached_response = folder / "response.json"
            if cached_response.exists():
                response = json.loads(cached_response.read_text())
                retries = response["retries"]
                state["wait_s"] = response["wall_s"]
                receive_only = response["receive_only_resume"]
            else:
                state.update(pending=True, not_forwarded=False)
                save(path, state)
                print(json.dumps({"question_id": qid, "stage": "waiting", "receive_only": receive_only,
                                  "prompt_chars": len(data["prompt"])}), flush=True)
                started = time.monotonic()
                try:
                    if receive_only:
                        response = asyncio.run(client.ask(bundle["base_url"], bundle["agent_id"], key,
                            data["prompt"], ref=data["ref"], history=state["history_id"], timeout=600,
                            receive_only=True))
                        retries = 0
                    else:
                        response, retries = client.ask_retrying(bundle["base_url"], bundle["agent_id"], key,
                            data["prompt"], ref=data["ref"], history=state["history_id"], timeout=600)
                except Exception as exc:
                    state["wait_s"] = state.get("wait_s", 0) + time.monotonic() - started
                    if isinstance(exc, client.Refused) and exc.forwarded is False:
                        state["not_forwarded"] = True
                    save(path, state)
                    error = {"type": type(exc).__name__, "message": str(exc).replace(key, "[REDACTED]"),
                        "code": getattr(exc, "code", None), "details": getattr(exc, "details", None),
                        "other_replies": getattr(exc, "other_replies", []),
                        "late_replies": getattr(exc, "late_replies", [])}
                    save(folder / "last-error.json", error)
                    print(json.dumps({"question_id": qid, "stage": "pending", "type": error["type"],
                                      "code": error["code"]}), flush=True)
                    return
                state["wait_s"] = state.get("wait_s", 0) + time.monotonic() - started
                save(folder / "response.json", {**response, "retries": retries, "wall_s": state["wait_s"],
                                                "receive_only_resume": receive_only})
            prediction, action, unknown = parse_prediction(response, data)
            result = {"question": data["question"], "prediction": prediction, "raw_final_action": action,
                "unknown_citations": unknown, "reactor_wall_s": state["wait_s"],
                "history_setup_s": state["history_setup_s"], "history_id": state["history_id"],
                "retries": retries, "receive_only_resume": receive_only}
            save(folder / "result.json", result)
            state["pending"] = False
            save(path, state)
            print(json.dumps({"stage": "answered", **prediction, "wall_s": state["wait_s"]}), flush=True)


def report(args):
    from score import read_jsonl, score_question
    gold = {g["question_id"]: g for g in read_jsonl(KIT.parent / "dev/gold.jsonl")}
    rows = []
    for qid in CASES:
        folder = args.out / qid
        if not (folder / "result.json").exists():
            continue
        old = json.loads((folder / "baseline.json").read_text())
        new = json.loads((folder / "result.json").read_text())
        before, after = [score_question(r["prediction"], gold[qid]) for r in (old, new)]
        rows.append({"question_id": qid, "question": new["question"]["question"],
            "before": {"prediction": old["prediction"], "score": before, "wall_s": old["reactor_wall_s"]},
            "after": {"prediction": new["prediction"], "score": after, "wall_s": new["reactor_wall_s"]},
            "citation_f1_delta": after["evidence_f1"] - before["evidence_f1"]})
    output = {"completed": len(rows), "planned": len(CASES), "cases": rows}
    if rows:
        for side in ("before", "after"):
            output[side + "_mean_f1"] = sum(r[side]["score"]["evidence_f1"] for r in rows) / len(rows)
    save(args.out / "comparison.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--agent-free", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    globals()[args.stage](args)
