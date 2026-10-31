"""Test fixed evidence across multiple questions without resetting a user's history.

Uses the same prompt builder, parser and receive-only timeout recovery as the
independent-case benchmark. Gold is read only by report(), never by freeze/run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parent))
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION
from evidence_packing import answer_message
from model_config import RETRIEVAL_PROFILE, require_current_profile
from search_backend import read_jsonl
from examples.benchmark_mcp import save, write_jsonl
from examples import benchmark_prompt_only as transport

DEFAULT = KIT / "benchmarks/long-multiturn-v5-20261031"


def checked_questions(root):
    require_current_profile(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["guidance_sha256"] != hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest():
        raise ValueError("Prompt changed since preparation")
    qs = read_jsonl(root / "questions.jsonl")
    if [q["question_id"] for q in qs] != protocol["cases"]:
        raise ValueError("Question order differs from the protocol")
    if len({q["question_id"] for q in qs}) != len(qs):
        raise ValueError("Duplicate question IDs")
    seen, previous = set(), None
    for q in qs:
        hid = q["history_id"]
        if hid != previous:
            if hid in seen:
                raise ValueError("Dataset histories must form contiguous groups")
            seen.add(hid)
            previous = hid
    return qs


def freeze(args):
    qs = checked_questions(args.out)
    plans = {p["question_id"]: p for p in read_jsonl(args.plans)}
    histories = {h["history_id"]: h for h in read_jsonl(args.out / "histories.jsonl")}
    if set(plans) != {q["question_id"] for q in qs}:
        raise ValueError("Plan/question mismatch")
    guidance_sha = hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest()
    positions = {}
    for q in qs:
        qid, hid = q["question_id"], q["history_id"]
        plan = plans[qid]
        assert plan["history_id"] == hid and plan["retrieval_profile"] == RETRIEVAL_PROFILE
        assert plan["question_sha256"] == hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()
        assert plan["guidance_sha256"] == guidance_sha
        valid = {s["session_id"] for s in histories[hid]["sessions"]}
        assert {e["id"] for e in plan["evidence"]} <= valid
        positions[hid] = positions.get(hid, 0) + 1
        folder = args.out / qid
        folder.mkdir(exist_ok=True)
        target = folder / "input.json"
        if target.exists():
            old = json.loads(target.read_text())
            assert old["question"] == q and old["evidence"] == plan["evidence"]
            continue
        ref = secrets.token_hex(4)
        prompt = answer_message(q, plan["evidence"], ref)
        assert len(prompt) == plan["message_chars"] <= 16000 and len(plan["evidence"]) <= 12
        save(target, {"question": q, "ref": ref, "prompt": prompt, "evidence": plan["evidence"],
                      "shown": list(range(len(plan["evidence"]))), "retrieval": plan,
                      "question_position_in_history": positions[hid]})
    print(json.dumps({"frozen": len(qs), "questions_per_history": positions}), flush=True)


def execute(root, questions):
    """Serial transport with a durable dataset-to-Reactor history binding."""
    registry_path = root / "histories-state.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        if list(root.glob("q_*/state.json")):
            raise ValueError("Missing history registry; recover it before continuing")
        registry = {"histories": {}, "active_dataset_history": None}
        save(registry_path, registry)
    for q in questions:
        if (root / "STOP").exists():
            return 3
        qid, hid = q["question_id"], q["history_id"]
        folder = root / qid
        state_path = folder / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if (folder / "result.json").exists():
            result = json.loads((folder / "result.json").read_text())
            if registry["histories"].get(hid) != result["history_id"]:
                raise ValueError("Completed result disagrees with history registry")
            continue
        if hid in registry["histories"]:
            if registry["active_dataset_history"] != hid:
                raise ValueError("Cannot return to a replaced Reactor history")
            remote = registry["histories"][hid]
            if state.get("history_id", remote) != remote:
                raise ValueError("Case checkpoint has a different Reactor history")
            if not state:
                save(state_path, {"creating_history": False, "history_id": remote,
                                  "history_setup_s": 0.0, "pending": False,
                                  "not_forwarded": False, "shared_history": True})
        else:
            # Refuse to replace any history with an unresolved request, including
            # when a caller accidentally passes a filtered questions list.
            for path in root.glob("q_*/state.json"):
                other = json.loads(path.read_text())
                if path != state_path and (other.get("pending") or other.get("creating_history")):
                    raise ValueError("Unresolved request must finish before creating another history")
            registry["creating_dataset_history"] = hid
            save(registry_path, registry)
        transport.CASES = (qid,)
        try:
            transport.run(SimpleNamespace(out=root, agent_free=True))
        finally:
            # Preserve a newly created remote ID even if answer parsing or waiting
            # fails. The per-question transport keeps ambiguous requests pending.
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("history_id"):
                remote = state["history_id"]
                if hid in registry["histories"] and registry["histories"][hid] != remote:
                    raise ValueError("Transport unexpectedly reset a dataset history")
                registry["histories"][hid] = remote
                registry["active_dataset_history"] = hid
                registry.pop("creating_dataset_history", None)
                save(registry_path, registry)
        if not (folder / "result.json").exists():
            return 3
        report(SimpleNamespace(out=root))
    return 0


def run(args):
    if not args.agent_free:
        raise SystemExit("Shared-agent authorization is required")
    import fcntl
    with (args.out / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args.out, checked_questions(args.out))


def report(args):
    from score import score_question, evidence_f1
    cases = {c["question_id"]: c for c in json.loads((args.out / "cases.json").read_text())}
    gold = ({g["question_id"]: g for g in read_jsonl(KIT.parent / "dev/gold.jsonl")}
            if any(c["split"] == "dev" for c in cases.values()) else {})
    rows, predictions = [], []
    for q in checked_questions(args.out):
        folder = args.out / q["question_id"]
        if not (folder / "result.json").exists():
            continue
        result = json.loads((folder / "result.json").read_text())
        data = json.loads((folder / "input.json").read_text())
        sent = sorted({e["id"] for e in data["evidence"]})
        row = {**cases[q["question_id"]], "question": q["question"], "dataset_history_id": q["history_id"],
               "reactor_history_id": result["history_id"], "position": data["question_position_in_history"],
               "prediction": result["prediction"], "reactor_wall_s": result["reactor_wall_s"],
               "history_setup_s": result["history_setup_s"], "retries": result["retries"],
               "passages": len(data["evidence"]), "message_chars": len(data["prompt"]),
               "sent_sessions": sent, "unknown_citations": result["unknown_citations"],
               "dev_local_score": None}
        if row["split"] == "dev":
            g = gold[q["question_id"]]
            row["dev_local_score"] = score_question(result["prediction"], g)
            row["retrieval_session_f1"] = evidence_f1(sent, g["evidence"]) if not g["abstain"] else None
        rows.append(row)
        predictions.append(result["prediction"])
    write_jsonl(args.out / "predictions.jsonl", predictions)
    save(args.out / "comparison.json", {"completed": len(rows), "planned": len(cases),
         "prompt_version": ANSWER_PROMPT_VERSION, "retrieval_profile": RETRIEVAL_PROFILE, "cases": rows,
         "limitations": "Test gold unavailable; source review is not an official answer judge. One shared-history sequence cannot establish a general latency trend."})
    print(json.dumps({"completed": len(rows), "planned": len(cases)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "run", "report"])
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--plans", type=Path)
    p.add_argument("--agent-free", action="store_true")
    args = p.parse_args()
    sys.exit(globals()[args.stage](args) or 0)
