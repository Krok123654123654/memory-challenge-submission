"""Prepare, resume and validate a Reactor-only challenge submission."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT.parent))
from score import validate as validate_format
from answer_guidance import ANSWER_GUIDANCE, ANSWER_PROMPT_VERSION
from evidence_packing import answer_message
from model_config import RETRIEVAL_PROFILE
from examples import benchmark_long_multiturn as runner
from examples.benchmark_mcp import save, write_jsonl
from search_backend import read_jsonl
from examples.benchmark_prompt_only import parse_prediction


def initialize(args):
    if (args.out / "protocol.json").exists():
        raise ValueError("Existing run: resume/check it; initialization never replaces checkpoints")
    qs = read_jsonl(args.data / "questions.jsonl")
    hs = read_jsonl(args.data / "histories.jsonl")
    groups = OrderedDict()
    for q in qs:
        groups.setdefault(q["history_id"], []).append(q)
    qs = [q for group in groups.values() for q in group]
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "questions.jsonl", qs)
    write_jsonl(args.out / "histories.jsonl", hs)
    save(args.out / "cases.json", [{"split": "test", "question_id": q["question_id"], "label": q["question"]} for q in qs])
    save(args.out / "protocol.json", {"cases": [q["question_id"] for q in qs], "histories": len(groups),
         "retrieval_profile": RETRIEVAL_PROFILE, "prompt_version": ANSWER_PROMPT_VERSION,
         "guidance": ANSWER_GUIDANCE, "guidance_sha256": hashlib.sha256(ANSWER_GUIDANCE.encode()).hexdigest(),
         "pruning_threshold": 0.05, "max_passages": 12, "full_message_limit": 16000,
         "history_policy": "One Reactor history per user; sequential questions; recover late replies without resets",
         "reactor_timeout_s": 600})
    print(json.dumps({"initialized_questions": len(qs), "histories": len(groups)}))


def check(root, only_history=None, require_complete_inputs=True):
    qs = runner.checked_questions(root)
    hs = {h["history_id"]: h for h in read_jsonl(root / "histories.jsonl")}
    if any(q["history_id"] not in hs for q in qs):
        raise ValueError("Question history missing")
    selected = [q for q in qs if only_history is None or q["history_id"] == only_history]
    if not selected:
        raise ValueError("No matching questions")
    missing = []
    for q in selected:
        p = root / q["question_id"] / "input.json"
        if not p.exists():
            missing.append(q["question_id"])
            continue
        data = json.loads(p.read_text())
        assert data["question"] == q, "Frozen question changed"
        assert data["prompt"] == answer_message(q, data["evidence"], data["ref"]), "Frozen message changed"
        assert len(data["prompt"]) <= 16000 and len(data["evidence"]) <= 12
        sessions = {s["session_id"]: s for s in hs[q["history_id"]]["sessions"]}
        for e in data["evidence"]:
            assert e["id"] in sessions and e["date"] == sessions[e["id"]]["date"], "Wrong source/date"
    if require_complete_inputs and missing:
        raise ValueError(f"{len(missing)} questions still need frozen evidence")
    return {"questions": len(qs), "histories": len({q['history_id'] for q in qs}),
            "checked_inputs": len(selected)-len(missing), "missing_inputs": len(missing),
            "completed": sum((root / q['question_id'] / 'result.json').exists() for q in qs)}


def validate_complete(root):
    readiness = check(root)
    qs = runner.checked_questions(root)
    hs = {h["history_id"]: h for h in read_jsonl(root / "histories.jsonl")}
    rows = read_jsonl(root / "predictions.jsonl")
    errors = validate_format(rows, {q["question_id"] for q in qs})
    if errors or len(rows) != len(qs):
        raise ValueError(f"Submission incomplete/invalid: {len(rows)}/{len(qs)}; {errors}")
    by_id = {r["question_id"]: r for r in rows}
    remote = {}
    unknown_labels = 0
    for q in qs:
        folder = root / q["question_id"]
        result = json.loads((folder / "result.json").read_text())
        state = json.loads((folder / "state.json").read_text())
        assert not state.get("pending")
        unknown_labels += len(result["unknown_citations"])
        assert by_id[q["question_id"]] == result["prediction"], "Prediction was changed after parsing Reactor"
        data = json.loads((folder / "input.json").read_text())
        response = json.loads((folder / "response.json").read_text())
        expected, _, _ = parse_prediction(response, data)
        assert expected == result["prediction"], "Prediction differs from the raw Reactor response"
        valid = {s["session_id"] for s in hs[q["history_id"]]["sessions"]}
        assert set(result["prediction"]["citations"]) <= valid
        remote.setdefault(q["history_id"], set()).add(result["history_id"])
    assert all(len(ids) == 1 for ids in remote.values()), "A user was reset mid-run"
    assert len(set.union(*remote.values())) == len(remote), "Two users shared Reactor memory"
    readiness.update(valid=True, unknown_labels_dropped=unknown_labels,
                     predictions_sha256=hashlib.sha256((root / "predictions.jsonl").read_bytes()).hexdigest())
    save(root / "validation.json", readiness)
    return readiness


def reparse_completed(root):
    """Deterministic format/citation parsing only; original replies stay untouched."""
    for q in runner.checked_questions(root):
        folder = root / q["question_id"]
        result_path = folder / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        response = json.loads((folder / "response.json").read_text())
        data = json.loads((folder / "input.json").read_text())
        prediction, action, unknown = parse_prediction(response, data)
        if (result["prediction"], result["raw_final_action"], result["unknown_citations"]) != (prediction, action, unknown):
            snapshot = folder / "result-before-parser-recovery.json"
            if not snapshot.exists():
                save(snapshot, result)
            result.update(prediction=prediction, raw_final_action=action, unknown_citations=unknown,
                          parsing_revision="json-delimiters-v2")
            save(result_path, result)
    runner.report(SimpleNamespace(out=root))


def run(args):
    if not args.agent_free:
        raise ValueError("Use --agent-free only when authorized to use the shared team agent")
    check(args.out, args.history)
    qs = runner.checked_questions(args.out)
    if args.history:
        qs = [q for q in qs if q["history_id"] == args.history]
    import fcntl
    awake = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        with (args.out / "batch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reparse_completed(args.out)
            last_pending, waits = None, 0
            while True:
                code = runner.execute(args.out, qs)
                if code == 0:
                    if not args.history:
                        print(json.dumps(validate_complete(args.out)), flush=True)
                    return 0
                if (args.out / "STOP").exists():
                    return code
                pending = [(q, json.loads((args.out / q["question_id"] / "state.json").read_text()))
                           for q in qs if (args.out / q["question_id"] / "state.json").exists()
                           and not (args.out / q["question_id"] / "result.json").exists()]
                if len(pending) != 1:
                    return code
                q, state = pending[0]
                error_path = args.out / q["question_id"] / "last-error.json"
                error = json.loads(error_path.read_text()) if error_path.exists() else {}
                if error.get("code") in {"history_changed", "history_required", "too_large", "unsupported"}:
                    return code
                # A new attempt after ambiguous delivery must stay receive-only.
                # Explicitly unforwarded busy requests may be retried by the client.
                if not state.get("pending") or (state.get("not_forwarded") and error.get("code") not in {"busy", "previous_pending"}):
                    return code
                waits = waits + 1 if last_pending == q["question_id"] else 1
                last_pending = q["question_id"]
                if waits > args.recovery_waits:
                    return code
                print(json.dumps({"stage": "recover_late_reply", "question_id": last_pending, "wait": waits}), flush=True)
                time.sleep(8)
    finally:
        if awake is not None:
            awake.terminate()


def package(args):
    if (KIT.parent / "upstream.lock.json").exists():
        target = args.out / "submission-code.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "archive", "--format=zip", "--output", str(target.resolve()), "HEAD"],
                       cwd=KIT.parent, check=True)
        print(json.dumps({"code_archive": str(target), "source": "committed public repository"}))
        return
    files = ["client.py", "answer_guidance.py", "evidence_packing.py", "model_config.py", "models.lock.json",
             "retrieval.py", "search_backend.py", "openprovence_backend.py", "submission.py", "requirements-retrieval.txt",
             "requirements-runtime.txt", "examples/benchmark_mcp.py", "examples/benchmark_prompt_only.py",
             "examples/benchmark_long_multiturn.py", "examples/benchmark_rerank_prune.py",
             "examples/prepare_openprovence.py", "examples/gpu_prepare_openprovence.py"]
    entries = {f"reactor/{n}": KIT / n for n in files}
    entries.update({"score.py": KIT.parent / "score.py", "README.md": KIT / "submission/README.md",
                    "PRESENTATION.md": KIT / "submission/PRESENTATION.md"})
    target = args.out / "submission-code.zip"
    hashes = {}
    for name, path in entries.items():
        data = path.read_bytes()
        if re.search(rb"\b(?:rxk|hf)_[A-Za-z0-9]{16,}\b", data):
            raise ValueError(f"Possible credential in {name}; archive not created")
        hashes[name] = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
    save(args.out / "code-manifest.json", {"files": hashes, "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
         "excluded": "credentials, test/dev data, model weights, benchmark artifacts, runtime history IDs"})
    print(json.dumps({"code_archive": str(target), "files": len(entries)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["init", "freeze", "check", "run", "validate", "package"])
    p.add_argument("--out", type=Path, default=KIT / "submission/final-20261031")
    p.add_argument("--data", type=Path)
    p.add_argument("--plans", type=Path)
    p.add_argument("--history")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--agent-free", action="store_true")
    p.add_argument("--recovery-waits", type=int, default=3)
    args = p.parse_args()
    if args.stage == "init":
        initialize(args)
    elif args.stage == "freeze":
        runner.freeze(args)
    elif args.stage == "check":
        print(json.dumps(check(args.out, args.history, not args.partial), indent=2))
    elif args.stage == "validate":
        print(json.dumps(validate_complete(args.out), indent=2))
    elif args.stage == "package":
        package(args)
    else:
        sys.exit(run(args))
