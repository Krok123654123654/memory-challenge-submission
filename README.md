# Memory Challenge — BGE + OpenProvence + Reactor

We retrieve and prune conversation evidence locally, then ask the organizer's Reactor agent for every final answer. The final configuration uses **BGE-large**, **BM25/RRF**, **OpenProvence at 0.05**, and the **v5 goal-based prompt**.

**Submission status:** partial results are available: 86/146 Reactor answers across 9/15 complete dataset histories. `predictions.jsonl` contains only completed answers, checked against saved Reactor replies. The remaining 60 questions are absent; no answers or abstentions were fabricated. See `validation.json` and `submission-status.json` for coverage. The run is still in progress. No official test accuracy is claimed.

## Reproduce

Use Python 3.12 for retrieval and Git for the organizer-client patch. Download the official test data from the [challenge releases](https://github.com/RxAI-dev/memory-challenge/releases). Keep `questions.jsonl` and `histories.jsonl` together in a local data directory.

```sh
python -m venv .venv
. .venv/bin/activate
python bootstrap.py
pip install -r reactor/requirements-runtime.txt -r reactor/requirements-retrieval.txt
cd reactor
python submission.py init --data /absolute/path/to/test --out /absolute/path/to/run
python examples/prepare_openprovence.py --input /absolute/path/to/run --out /absolute/path/to/retrieval --device cuda --save-audit
python submission.py freeze --out /absolute/path/to/run --plans /absolute/path/to/retrieval/plans.jsonl
python submission.py concise --out /absolute/path/to/run --schedule ../run-settings.json
python submission.py check --out /absolute/path/to/run
python submission.py run --out /absolute/path/to/run --agent-free
python submission.py validate --out /absolute/path/to/run
```

Choose `mps` or `cpu` instead of `cuda` where necessary. Retrieval can be prepared on rented GPU hardware with `examples/gpu_prepare_openprovence.py`; only our own models run there. The final run's complete retrieval preparation took about 333 seconds on an A10G. Reactor answering remains sequential and can take minutes per question. Embedding caches and frozen inputs are retained locally.

For a small organizer rerun, supply a subset of questions with their full source histories in a new data directory and follow the same commands. Gold is never used for retrieval or answering. The files named `benchmark_*` supply shared adapters and transport; `submission.py` is the final entry point. The final run uses fixed retrieved evidence, without model-controlled MCP search. Questions 42–50 used concise output instructions at the user’s request. From question 51 onward, the original v5 output instructions were restored. The retry backoff remains capped at 5 seconds. Full v5 guidance, retrieved evidence, existing answers and user histories were preserved throughout. `run-settings.json` records the per-question output style so a rerun uses the same prompts.

## Credentials and safe resume

Store the organizer's JSON bundle (`base_url`, `agent_id`, `access_key`) in `~/.config/reactor/credentials.json`, with permission `0600`. Enter the secret locally; never commit it or put it in a command/chat. `--agent-free` asserts that the team has authorized exclusive use of its single shared Reactor agent.

The same run command resumes checkpoints. It preserves **one Reactor history per dataset history**, processes that user's questions sequentially, and resets only when moving to another user. The 600-second timeout does not cancel a remote request. Ambiguous pending requests are recovered by listening in the original history; only explicitly unforwarded busy requests may be sent again. Three further receive-only waits are allowed before stopping for attention.

Create `STOP` in the run directory to stop after the current question. Remove it only to deliberately resume. Do not delete checkpoints or reset memory to recover a timeout. On macOS the run keeps the computer awake while it is active.

## Retrieval and answer contract

- Index role-labelled fragments of about 140 words with 100-word stride and neighboring context; embedding windows are capped at 480 tokens with overlap.
- Fuse BGE-large and BM25 rankings using RRF with `k=60`; rank/prune the top 40 fragments using the pinned OpenProvence model.
- Pack up to 12 passages within the **complete 16,000-character message**. Preserve speaker roles, session IDs, dates and day offsets to the question date.
- Give Reactor the v5 role, goal and evaluation context. Map its final-answer `[E#]` labels to supplied session IDs. Unknown labels are recorded and excluded; no guessed citation fallback is used.
- Preserve raw replies. Parsing may close missing JSON delimiters or infer a missing `abstain` flag from the explicit insufficient-evidence marker, but never writes answer facts. Incomplete requests are never fabricated as abstentions.

Validation checks every expected question exactly once, the required schema, valid session IDs, completed states, one distinct remote history per user, and that each prediction equals a fresh parse of its saved Reactor reply.

## Submission files

1. `predictions.jsonl` — saved Reactor answers; current coverage is recorded in `submission-status.json`.
2. This public repository and its short run instructions.
3. `presentation/solution.pptx` and [speaker outline](PRESENTATION.md).

Private run state, source histories, credentials, caches and model weights are excluded from Git. Model revisions are in `reactor/models.lock.json`.
