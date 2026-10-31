# Case assistant

A local web app for reviewing a case file, for prosecution or defence. You **chat** with the case: every answer cites
the documents it comes from, and clicking a citation opens the document at that line. Questions can be about the whole
case or **one person**, can name a **period** ("what happened between 1 and 31 August 2025?", "on 14 March"), and a
**did-it-happen** question gets a Yes, No or Not in file. A **timeline** lists every dated event, the **people** view
shows what each person said and what others said about them, and the **documents** view reads statements and
transcripts. When the file does not contain the answer, it says so.

It finds, orders and cites. It does not assess guilt, credibility or risk; a person checks every answer against
its sources.

## Run it

Needs Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
. .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # Linux without a GPU: first pip install torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env                     # then add the Reactor key, or an alternative's (below)
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, pick a case at the top and ask in the chat. The first start downloads the sentence
embedder (BAAI/bge-small-en-v1.5, about 130 MB) and takes a little longer. The server listens on this machine only.

To check that search finds the right documents, without any model:

```sh
python scripts/check_case.py cases/calder-freight     # also cases/marlow-street, cases/okafor-deposit
```

## Answers

**Reactor is the main answer mode.** OpenRouter and a local model are alternatives, for when Reactor is unavailable
or too slow, or to keep everything on this machine. Pick the mode in the **Answers** menu at the top. Keys go in
`.env`; the server reads it again on each request, so a key added there works without a restart. A mode without its
settings is greyed out.

| Mode | Settings in `.env` | What it does | Data leaves the machine? |
| --- | --- | --- | --- |
| **Reactor** (main, the default) | `REACTOR_BASE_URL`, `REACTOR_AGENT_ID`, `REACTOR_ACCESS_KEY` | written, cited answers from the hackathon's hosted agent, one Reactor history per chat; about a minute per answer | yes |
| OpenRouter (alternative) | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default `openai/gpt-6-luna`) | the same through OpenRouter, in a few seconds | yes: OpenRouter and the model provider |
| Local model (alternative) | `LOCAL_LLM_URL`, `LOCAL_LLM_MODEL` | the same from any OpenAI-compatible server you run (llama.cpp, vLLM, Ollama) | no |
| Sources only | none | the most relevant passages, cited, in order | no |

The app opens with Reactor when it is configured; `ANSWERS_DEFAULT` in `.env` (`reactor`, `openrouter`, `local` or
`retrieval`) changes that. Use the hosted modes with the synthetic cases only, never with real case files. Every
question, the passages used and the answer are appended to the case's `audit.jsonl`. A *did it happen* question gets a
**Yes**, **No**, **Disputed** (the sources contradict each other) or **Not in file** badge.

## Chat

Each chat keeps its own conversation (a Reactor history, or the earlier turns sent along to OpenRouter or a local
model), so follow-ups work ("and his brother?"); a follow-up is also searched together with the previous question.
*New chat* starts afresh. A period named in a message narrows the search to
passages about events in it, in the order they happened, and the model is told the period. Each passage's event date
is the date its text names (or the last one before it in the same statement or interview), not only the date the
document was written.

## A case

```
cases/<name>/
  case.json            # optional: title, type, reference, summary
  people.json          # id, name, role, aliases, notes
  documents/*.txt|md|pdf|docx
  questions.jsonl      # optional: questions with known answers, for scripts/check_case.py
```

A text document may start with `title:`, `kind:` (statement, interview, messages, email, report), `date:` and
`author:` (a person id) lines, then `---`. In transcripts and message logs, a line starting with a person's name or
alias and a colon is that person speaking. A person is linked to what they said and to every passage that names them
(single-word aliases match case-sensitively: "Price" the person, not "price").

Three synthetic cases (fictional people and events), each with deliberate traps:

| Case | Kind | Traps |
| --- | --- | --- |
| `marlow-street` | criminal: theft from a gallery store | a witness who corrects her time, hearsay, two Martins, a brother's van, phone data against an alibi |
| `calder-freight` | civil: financial fraud | invoices split under the approval limit, kickbacks to the director's company, an approval sent from the CEO's mailbox while he was on leave, a clerk who corrects a date, two Webers |
| `okafor-deposit` | civil: tenancy deposit | mould reported for months before the fan was fixed, deductions for damage already on the check-in inventory |

```sh
python scripts/check_case.py cases/calder-freight   # are the expected documents found? (13/13, 9/9, 5/5)
```

## How it searches

BM25 and a sentence embedder (BAAI/bge-small-en-v1.5, MIT) over passages of a few sentences, rankings fused by
reciprocal rank fusion plus each ranking's own best hits, so a passage that shares no word with the question is still
found. For one person, only the passages they said or that name them. The approach was measured on the RxAI Memory
Retrieval Challenge.

## Limits

- People and aliases come from `people.json`; relations ("my brother") are not resolved automatically.
- No contradiction view yet; the prompt asks the model to flag conflicting accounts and cite each version.
- In the EU, AI that evaluates evidence in criminal matters is high-risk under the AI Act: keep a person in the loop,
  keep the audit log, and use it for search and citation, not assessment.
