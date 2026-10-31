"""Paired Reactor experiment: fixed retrieval versus model-controlled local MCP search.

Reactor's message API is adapted to MCP by this host; no native Reactor MCP support
is claimed. Prepare randomly samples dev without gold; only report reads gold.
Run only when authorized to take the shared agent: each independent trial starts
a fresh history, and tool rounds never reset it. Pending requests are checkpointed.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import statistics
import subprocess
import sys
import time

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT.parent))
import client
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION, SEARCH_GUIDANCE
from search_backend import SearchBackend, TOOLS, read_jsonl
from evidence_packing import MAX_PASSAGES
from model_config import RETRIEVAL_PROFILE, RERANKER_PIN, require_current_profile

DEFAULT = KIT / "benchmarks/mcp-bge-20261031"


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    previous = set()
    for p in [KIT / "benchmarks/as-is-20261031/questions.json",
              KIT / "benchmarks/pilot-20261031/sample.json"]:
        if p.exists():
            previous.update((r.get("question", r) if isinstance(r.get("question"), dict) else r)["question_id"]
                            for r in json.loads(p.read_text()))
    pool = [q for q in read_jsonl(KIT.parent / "dev/questions.jsonl") if q["question_id"] not in previous]
    selected = random.Random(args.seed).sample(sorted(pool, key=lambda q:q["question_id"]), args.count)
    assert len({q["history_id"] for q in selected}) == len(selected), "This experiment uses dev's distinct histories"
    write_jsonl(args.out / "questions.jsonl", selected)
    chosen = {q["history_id"] for q in selected}
    write_jsonl(args.out / "histories.jsonl", [h for h in read_jsonl(KIT.parent / "dev/histories.jsonl") if h["history_id"] in chosen])
    # Counterbalance which method goes first; every pair has the same question.
    order = [i % 2 for i in range(len(selected))]
    random.Random(args.seed+1).shuffle(order)
    trials = []
    for q, first in zip(selected, order):
        for mode in (["fixed", "mcp"] if first == 0 else ["mcp", "fixed"]):
            trials.append({"question_id": q["question_id"], "mode": mode,
                           "trial_id": q["question_id"] + "-" + mode})
    save(args.out / "protocol.json", {"seed": args.seed, "count": args.count,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "sampling": "uniform random dev questions, excluding previously evaluated 84 question IDs; no gold read",
        "comparison": "same initial Arctic+BM25 top40 / MIT BGE rerank-only / original chunks / at most12 passages / exact16000-character message budget",
        "date_annotations": "calendar-day offset from session date to question date; original dates retained; no inferred event dates",
        "mcp": "real stdio MCP 2025-06-18 server and host adapter translating Reactor JSON actions; not native Reactor tool calls",
        "tools": [t["name"] for t in TOOLS], "max_tool_rounds": 2, "max_calls_per_round": 3,
        "tool_arm": "first turn is search planning only; then answer using tool evidence, with at most one further search round",
        "tool_limits": "same user's history only; inclusive chat-date filters before top-k; session ID allowlist",
        "history_policy": "independent benchmark trials; one fresh Reactor history per dataset history per method; no reset inside a trial",
        "latency": "wall latency includes Reactor turns, internal busy retries and dynamic search; initial retrieval reported separately",
        "reranker": RERANKER_PIN,
        "trials": trials, "status": "prepared"})
    save(args.out / "source-hashes.json", {str(p.relative_to(KIT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [KIT / "search_backend.py", Path(__file__), KIT / "examples/serve_history_mcp.py",
                  KIT / "client.py", KIT / "answer_guidance.py"]})
    print(json.dumps({"selected": [q["question_id"] for q in selected], "trials": len(trials)}), flush=True)


def index(args):
    backend = SearchBackend(args.out)
    target = args.out / "initial-plans.jsonl"
    done = {r["question_id"]:r for r in read_jsonl(target)} if target.exists() else {}
    for q in backend.questions.values():
        if q["question_id"] in done:
            continue
        result = backend.search_history(q["question_id"], q["question"], limit=MAX_PASSAGES)
        done[q["question_id"]] = {"question_id":q["question_id"], **result}
        write_jsonl(target, done.values())
        print(json.dumps({"indexed":q["question_id"], "passages":len(result["evidence"]), "search_s":result["seconds"]}), flush=True)


class MCPClient:
    def __init__(self, root):
        self.log = (root / "mcp-server.log").open("a")
        self.proc = subprocess.Popen([sys.executable, str(KIT / "examples/serve_history_mcp.py"), "--run-dir", str(root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", bufsize=1)
        self.counter = 0
        result = self.request("initialize", {"protocolVersion":"2025-06-18", "capabilities":{},
            "clientInfo":{"name":"reactor-benchmark-host", "version":"1.0.0"}})
        assert result["protocolVersion"] == "2025-06-18"
        self.proc.stdin.write(json.dumps({"jsonrpc":"2.0", "method":"notifications/initialized"})+"\n")
        self.proc.stdin.flush()
        self.tools = self.request("tools/list", {})["tools"]

    def request(self, method, params):
        self.counter += 1
        self.proc.stdin.write(json.dumps({"jsonrpc":"2.0", "id":self.counter, "method":method, "params":params})+"\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server exited; see mcp-server.log")
        data = json.loads(line)
        if data.get("id") != self.counter or "error" in data:
            raise RuntimeError(f"MCP response error: {data}")
        return data["result"]

    def call(self, qid, name, arguments):
        if name not in {t["name"] for t in TOOLS} or not isinstance(arguments, dict):
            raise ValueError("Unknown tool or invalid arguments")
        if "question_id" in arguments and arguments["question_id"] != qid:
            raise ValueError("The active user and question cannot be changed")
        args = {**arguments, "question_id":qid}
        if name == "search_history":
            args.setdefault("limit", 4)
        result = self.request("tools/call", {"name":name,"arguments":args})
        if result.get("isError"):
            raise ValueError(result["content"][0]["text"])
        return json.loads(result["content"][0]["text"])

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.terminate(); self.proc.wait(timeout=15)
        self.proc.stdout.close()
        self.log.close()


def add_evidence(state, passages):
    priority = []
    for passage in passages:
        found = next((i for i,e in enumerate(state["evidence"]) if e == passage), None)
        if found is None:
            found = len(state["evidence"])
            state["evidence"].append(passage)
        priority.append(found)
    state["priority"] = list(dict.fromkeys(priority + state.get("priority", [])))


def prompt_for(q, state, ref, allow_tools):
    if state["mode"] == "fixed":
        from evidence_packing import answer_message
        ordered = [state["evidence"][i] for i in state["priority"]]
        if state["priority"] != list(range(len(ordered))):
            raise ValueError("Fixed evidence labels must follow packed order")
        if len(ordered) > MAX_PASSAGES:
            raise ValueError("Too many evidence passages")
        result = answer_message(q, ordered, ref)
        if len(result) > client.MAX_CHARS:
            raise ValueError("Prompt exceeds client character budget")
        state["shown"] = list(state["priority"])
        return result
    planning_only = state["mode"] == "mcp" and state["tool_rounds"] == 0
    intro = (f"[ref:{ref}] Question: {q['question']}\n"
        f"Question date: {q['question_date']}. Treat that date as now for this question.\n"
        "The excerpts below were retrieved using an embedding model and keyword search, then ranked for relevance.\n"
        + ANSWER_GUIDANCE + "\n")
    instructions = ('\nReturn JSON: {"action":"answer","answer":"your answer [E1]","abstain":false}. '
        'Place all supporting [E#] labels inside the "answer" string; labels elsewhere are not collected. '
        'For abstention set abstain:true; use "INSUFFICIENT_EVIDENCE" followed by labels for genuine partial '
        'evidence, or an empty answer if none. The host submits an empty abstention answer and retains those citations.\n')
    if planning_only:
        intro = (f"[ref:{ref}] Task: Choose history-search tool calls for a later answering step.\n"
            f"Target question: {q['question']}\nQuestion date: {q['question_date']}. Treat that date as now.\n"
            + SEARCH_GUIDANCE + "\n")
        instructions = "\nThis retrieval turn requires a tools action containing at least one search_history call.\n"
    if allow_tools:
        schemas = copy.deepcopy(TOOLS)
        for t in schemas:
            t["inputSchema"]["properties"].pop("question_id", None)
            t["inputSchema"]["required"].remove("question_id")
        instructions += ("The local search tools retrieve further evidence. "
            "The host fixes the user and question; date filters refer to chat dates, not dates of events described. "
            f"You have {2-state['tool_rounds']} search rounds left, at most 3 tool calls per round. "
            'To use tools, return {"action":"tools","calls":[{"name":"search_history","arguments":{"query":"search phrases"}}]}. '
            "The host will execute them and return evidence in this same conversation.\nTools: " + json.dumps(schemas) + "\n")
    elif state["mode"] == "mcp":
        instructions += "No more tool rounds are available; answer from the evidence already received, or abstain.\n"
    notes = "\nTool results: " + json.dumps(state.get("tool_notes", []), ensure_ascii=False) + "\n"
    ending = f"\nWrite the JSON on one line starting exactly with # FINAL ANSWER [ref:{ref}]:\nThis marker transports your response; a tools action is intermediate and is not submitted as an answer."
    lines, used, shown = [], 0, []
    for i in state["priority"]:
        e = state["evidence"][i]
        line = f"[E{i+1}] (source: {e['id']}, date: {e['date']}) {e['text']}"
        if len(lines) >= MAX_PASSAGES:
            break
        candidate = intro + "\nEvidence:\n" + "\n".join(lines + [line]) + notes + instructions + ending
        if len(candidate) > client.MAX_CHARS:
            continue
        lines.append(line); used += len(line); shown.append(i)
    state["shown"] = sorted(set(state.get("shown", [])) | set(shown))
    result = intro + "\nEvidence:\n" + "\n".join(lines) + notes + instructions + ending
    if len(result) > client.MAX_CHARS:
        raise ValueError("Prompt exceeds client character budget")
    return result


def parse_action(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except ValueError:
        # Reactor sometimes omits the final JSON quote/brace. Close syntax only;
        # never supply answer words or a missing substantive claim.
        if cleaned.startswith('{'):
            for suffix in ('"}', '}'):
                try:
                    recovered = json.loads(cleaned + suffix)
                except ValueError:
                    continue
                if (isinstance(recovered, dict) and recovered.get("action") == "answer"
                        and isinstance(recovered.get("answer"), str)
                        and ("abstain" not in recovered or type(recovered["abstain"]) is bool)):
                    recovered.setdefault("abstain", bool(client._INSUFFICIENT.match(recovered["answer"])))
                    recovered["format_deviation"] = True
                    recovered["closed_json_delimiters"] = suffix
                    return recovered
        # Preserve an ordinary Reactor final answer verbatim, and record the deviation.
        return {"action":"answer", "answer":text.strip(),
                "abstain":bool(client._INSUFFICIENT.match(text)), "format_deviation":True}
    if not isinstance(value, dict) or value.get("action") not in ("answer", "tools"):
        raise ValueError("Invalid action object")
    if value["action"] == "answer":
        if isinstance(value.get("answer"), str) and "abstain" not in value:
            value["abstain"] = bool(client._INSUFFICIENT.match(value["answer"]))
            value["format_deviation"] = True
        if not isinstance(value.get("answer"), str) or type(value.get("abstain")) is not bool:
            raise ValueError("Answer requires text and a Boolean abstain field")
    elif not isinstance(value.get("calls"), list) or not 1 <= len(value["calls"]) <= 3:
        raise ValueError("A tools action requires 1 to 3 calls")
    return value


def run(args):
    require_current_profile(args.out)
    protocol = json.loads((args.out / "protocol.json").read_text())
    if protocol.get("answer_prompt_version") != ANSWER_PROMPT_VERSION:
        raise RuntimeError("This experiment used an older prompt. Prepare a separate --out directory for "
                           "the revised prompt; do not mix old results or pending requests with a new version.")
    os.umask(0o077)
    import fcntl
    lock = (args.out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundle_path = Path.home() / ".config/reactor/credentials.json"
    assert bundle_path.stat().st_mode & 0o077 == 0
    bundle = json.loads(bundle_path.read_text())
    key = bundle["access_key"]
    questions = {q["question_id"]:q for q in read_jsonl(args.out / "questions.jsonl")}
    plans = {p["question_id"]:p for p in read_jsonl(args.out / "initial-plans.jsonl")}
    assert set(questions) == set(plans)
    if any(p.get("retrieval_profile") != RETRIEVAL_PROFILE for p in plans.values()):
        raise ValueError("Cached plans belong to another retrieval pipeline; prepare a fresh experiment")
    mcp = None
    try:
        if not (args.out / "STOP").exists():
            mcp = MCPClient(args.out)
            warm_q = next(iter(questions.values()))
            t = time.perf_counter()
            warm = mcp.call(warm_q["question_id"], "search_history", {"query":warm_q["question"], "limit":MAX_PASSAGES})
            assert warm["evidence"] == plans[warm_q["question_id"]]["evidence"], "Fixed and MCP retrieval must agree"
            save(args.out / "mcp-warmup.json", {"seconds":time.perf_counter()-t,
                 "excluded_from_query_latency":True,"initial_evidence_equivalence_verified":True})
        for trial in protocol["trials"]:
            if (args.out / "STOP").exists():
                print("Stopped between requests; Reactor memory preserved.", flush=True); return
            folder = args.out / trial["trial_id"]
            folder.mkdir(exist_ok=True)
            if (folder / "result.json").exists():
                continue
            q = questions[trial["question_id"]]
            state_path = folder / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
            else:
                state = {**trial, "evidence":[], "priority":[], "shown":[], "steps":[],
                         "tool_rounds":0, "tool_events":[], "pending":None, "started_epoch":time.time()}
                add_evidence(state, plans[q["question_id"]]["evidence"])
            if "reactor_history" not in state:
                if state.get("creating_history"):
                    raise RuntimeError("Interrupted history creation; recover its ID before resuming")
                state["creating_history"] = True
                save(state_path,state)
                t = time.perf_counter()
                history = client.new_history(bundle["base_url"],bundle["agent_id"],key,timeout=600)
                state.update(reactor_history=history["history_id"],creating_history=False,history_setup_s=time.perf_counter()-t)
                save(state_path,state)
                print(json.dumps({"trial":trial["trial_id"],"stage":"fresh_history","question":q["question"]}),flush=True)
            while True:
                if (args.out / "STOP").exists():
                    print("Stopped between requests; checkpoint retained.",flush=True); return
                # A persisted response is processed once without asking Reactor again.
                if state.get("unprocessed") is not None:
                    response = state.pop("unprocessed")
                else:
                    resume = state["pending"] is not None and not state["pending"].get("not_forwarded", False)
                    if state["pending"] is None:
                        ref = secrets.token_hex(4)
                        prompt = prompt_for(q,state,ref,state["mode"]=="mcp" and state["tool_rounds"]<2)
                        state["pending"] = {"ref":ref,"prompt":prompt,"started_epoch":time.time()}
                        save(state_path,state)
                    pending = state["pending"]
                    print(json.dumps({"trial":trial["trial_id"],"stage":"waiting","round":len(state["steps"])+1,
                                      "chars":len(pending["prompt"]),"receive_only":resume}),flush=True)
                    t = time.perf_counter()
                    try:
                        if resume:
                            response = asyncio.run(client.ask(bundle["base_url"],bundle["agent_id"],key,pending["prompt"],
                                timeout=600,ref=pending["ref"],history=state["reactor_history"],receive_only=True))
                            retries = 0
                        else:
                            response,retries = client.ask_retrying(bundle["base_url"],bundle["agent_id"],key,pending["prompt"],
                                timeout=600,ref=pending["ref"],history=state["reactor_history"])
                    except Exception as exc:
                        if isinstance(exc, client.Refused) and exc.forwarded is False:
                            state["pending"]["not_forwarded"] = True
                            save(state_path,state)
                        error = {"type":type(exc).__name__,"message":str(exc).replace(key,"[REDACTED]"),
                                 "code":getattr(exc,"code",None),"pending_ref":pending["ref"]}
                        save(folder / "last-error.json",error)
                        print(json.dumps({"trial":trial["trial_id"],"stage":"stopped_pending",**error}),flush=True)
                        return
                    step = {**pending,**response,"wall_s":time.perf_counter()-t,"retries":retries,"receive_only":resume}
                    state["steps"].append(step)
                    state["pending"] = None
                    state["unprocessed"] = response
                    save(state_path,state)
                    response = state.pop("unprocessed")
                action = parse_action(response["answer"])
                if action["action"] == "answer":
                    answer = action["answer"]
                    cited,unknown = client.citations(answer,state["evidence"])
                    allowed_labels = {f"E{i+1}" for i in state["shown"]}
                    valid = [c for c in cited if c["label"] in allowed_labels]
                    unknown += [c["label"] for c in cited if c["label"] not in allowed_labels]
                    abstain = action["abstain"] or bool(client._INSUFFICIENT.match(answer))
                    cleaned = re.sub(r"\[(?:E\d+[\s,]*)+\]", "", answer).strip()
                    prediction = {"question_id":q["question_id"],"answer":"" if abstain else cleaned,
                                  "abstain":abstain,"citations":list(dict.fromkeys(c["id"] for c in valid))}
                    result = {**trial,"question":q,"prediction":prediction,"raw_final_action":action,
                        "answer_prompt_version":ANSWER_PROMPT_VERSION,
                        "reactor_history":state["reactor_history"],"history_setup_s":state["history_setup_s"],
                        "reactor_wall_s":sum(s["wall_s"] for s in state["steps"]),
                        "initial_search_s":plans[q["question_id"]]["seconds"],
                        "dynamic_tool_wall_s":sum(e["wall_s"] for e in state["tool_events"]),
                        "elapsed_s":time.time()-state["started_epoch"],"reactor_turns":len(state["steps"]),
                        "tool_rounds":state["tool_rounds"],"tool_events":state["tool_events"],
                        "unknown_citations":unknown,"shown_evidence":[state["evidence"][i] for i in state["shown"]]}
                    save(folder / "result.json",result); save(state_path,state)
                    print(json.dumps({"trial":trial["trial_id"],"stage":"answered","prediction":prediction,
                                      "reactor_wall_s":result["reactor_wall_s"],"tool_rounds":state["tool_rounds"]},ensure_ascii=False),flush=True)
                    break
                if state["mode"] != "mcp" or state["tool_rounds"] >= 2:
                    raise RuntimeError("Reactor requested tools outside the allowed tool budget; response preserved")
                if mcp is None:
                    mcp = MCPClient(args.out)
                notes = []
                for call in action["calls"]:
                    t = time.perf_counter()
                    try:
                        result = mcp.call(q["question_id"],call.get("name"),call.get("arguments",{}))
                        add_evidence(state,result.get("evidence",[]))
                        note = {"tool":call["name"],"arguments":call.get("arguments",{}),
                                **{k:v for k,v in result.items() if k not in ("evidence","candidate_sessions")}}
                        status = "ok"
                    except ValueError as exc:
                        result = {"error":str(exc)}
                        note = {"tool":call.get("name"),**result}; status = "error"
                    event = {"call":call,"result":result,"status":status,"wall_s":time.perf_counter()-t}
                    state["tool_events"].append(event); notes.append(note)
                    print(json.dumps({"trial":trial["trial_id"],"stage":"tool","name":call.get("name"),
                                      "arguments":call.get("arguments",{}),"status":status,"wall_s":event["wall_s"]}),flush=True)
                state["tool_rounds"] += 1
                state["tool_notes"] = notes
                save(state_path,state)
        protocol["status"] = "complete"
        save(args.out / "protocol.json",protocol)
    finally:
        if mcp is not None:
            mcp.close()
        lock.close()


def report(args):
    from score import score_question, summarize, evidence_f1
    gold = {g["question_id"]:g for g in read_jsonl(KIT.parent / "dev/gold.jsonl")}
    protocol = json.loads((args.out / "protocol.json").read_text())
    results = [json.loads(p.read_text()) for p in args.out.glob("q_*/result.json")]
    output = {"planned_cases":protocol["count"],"completed_trials":len(results),"methods":{}}
    details = []
    for mode in ("fixed","mcp"):
        rows = [r for r in results if r["mode"]==mode]
        if not rows:
            continue
        scored = [score_question(r["prediction"],gold[r["question_id"]]) for r in rows]
        write_jsonl(args.out / f"{mode}-predictions.jsonl",[r["prediction"] for r in rows])
        output["methods"][mode] = {"score":summarize(scored),
            "median_reactor_wall_s":statistics.median(r["reactor_wall_s"] for r in rows),
            "mean_total_work_s":statistics.mean(r["reactor_wall_s"]+r["dynamic_tool_wall_s"]+r["initial_search_s"] for r in rows),
            "total_reactor_turns":sum(r["reactor_turns"] for r in rows),
            "cases_using_tools":sum(r["tool_rounds"]>0 for r in rows),
            "tool_errors":sum(e["status"]=="error" for r in rows for e in r["tool_events"]),
            "calls_with_date_filters":sum(bool(e["call"].get("arguments",{}).get("date_from") or e["call"].get("arguments",{}).get("date_to")) for r in rows for e in r["tool_events"]),
            "calls_with_session_filters":sum("session_ids" in e["call"].get("arguments",{}) for r in rows for e in r["tool_events"])}
        for r,s in zip(rows,scored):
            g = gold[r["question_id"]]
            details.append({**s,"mode":mode,"question":r["question"]["question"],"prediction":r["prediction"],
                            "gold_answer":g["answer"],"gold_evidence":g["evidence"],
                            "retrieved_session_f1":evidence_f1([e["id"] for e in r["shown_evidence"]],g["evidence"]) if not g["abstain"] else None,
                            "reactor_wall_s":r["reactor_wall_s"],"tool_rounds":r["tool_rounds"]})
    output["limitations"] = ["Five-case exploratory sample, not reliable evidence of a general winner",
        "Unmatched answer strings remain unjudged under the official local scorer; not automatically wrong",
        "This tests model-controlled retrieval with an MCP host adapter, not native Reactor MCP or protocol overhead",
        "Dev has one question per history; does not measure the test split's accumulating memory",
        "Both variants use dates; this does not isolate the effect of relative-date annotations"]
    save(args.out / "comparison.json",output)
    save(args.out / "scored-details.json",details)
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["prepare","index","run","report"])
    ap.add_argument("--out",type=Path,default=DEFAULT)
    ap.add_argument("--seed",type=int,default=20261031)
    ap.add_argument("--count",type=int,default=5)
    args = ap.parse_args()
    globals()[args.stage](args)
