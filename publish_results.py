"""Publish only a complete, validated run. Never upload private checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reactor"))
import submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--wait-for-pid", type=int)
    p.add_argument("--push", action="store_true", help="Publish to the already-configured origin")
    args = p.parse_args()
    questions = submission.runner.checked_questions(args.run)
    expected = len(questions)
    while True:
        complete = sum((args.run / q["question_id"] / "result.json").exists() for q in questions)
        if complete == expected:
            # Wait for the runner's final validation as well as all result files.
            if (args.run / "validation.json").exists():
                break
        if not args.wait_for_pid:
            raise SystemExit(f"Not publishing: {complete}/{expected} answers; complete validation is required")
        try:
            os.kill(args.wait_for_pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"Runner stopped at {complete}/{expected}; nothing was published")
        time.sleep(30)
    validation = submission.validate_complete(args.run)
    payload = (args.run / "predictions.jsonl").read_bytes()
    if re.search(rb"\b(?:hf|rxk|ghp|gho)_[A-Za-z0-9]{16,}\b", payload):
        raise SystemExit("Possible credential in answers; stopped before publishing")
    # A team access key is never meant to be an answer. Do not log it.
    credentials = Path.home() / ".config/reactor/credentials.json"
    if credentials.exists():
        key = json.loads(credentials.read_text()).get("access_key", "")
        if key and key.encode() in payload:
            raise SystemExit("Private credential detected; stopped before publishing")
    (ROOT / "predictions.jsonl").write_bytes(payload)
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    timings = [json.loads((args.run / q["question_id"] / "result.json").read_text())["reactor_wall_s"]
               for q in questions]
    status = {"status": "complete", "questions": expected, "histories": validation["histories"],
              "previous_answers_reused": 0, "prompt_version": submission.ANSWER_PROMPT_VERSION,
              "retrieval_profile": submission.RETRIEVAL_PROFILE,
              "total_answer_wait_s": round(sum(timings), 2),
              "predictions_sha256": hashlib.sha256(payload).hexdigest(),
              "completed_utc": datetime.now(timezone.utc).isoformat(), "official_score": "not available"}
    protocol = json.loads((args.run / "protocol.json").read_text())
    status["speed_adjustment"] = protocol.get("speed_adjustment")
    from collections import Counter
    status["response_style_counts"] = dict(Counter(protocol.get("response_styles", {}).values()))
    (ROOT / "submission-status.json").write_text(json.dumps(status, indent=2) + "\n")
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = re.sub(r"\*\*Submission status:\*\*.*?\n\n",
                  "**Submission status:** all 146 fresh answers are complete and validated. "
                  "See `predictions.jsonl`, `validation.json` and `submission-status.json`. "
                  "No previous experimental answers were reused. Official test scores remain with the organizers.\n\n",
                  text, count=1, flags=re.S)
    text = text.replace("`predictions.jsonl` — published after the fresh run finishes and validates.",
                        "`predictions.jsonl` — 146 complete, validated Reactor answers.")
    readme.write_text(text)
    if args.push:
        subprocess.run(["git", "add", "predictions.jsonl", "validation.json", "submission-status.json", "README.md"], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", "Publish 146 validated fresh Reactor predictions"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
