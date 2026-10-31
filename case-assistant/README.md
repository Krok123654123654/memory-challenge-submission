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

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The server listens on this machine only.

## Answers

| Mode | What it does | Data leaves the machine? |
| --- | --- | --- |
| Reactor (demo; the default when configured) | a written, cited answer from the hackathon's hosted agent, one Reactor history per chat; synthetic data only | yes |
| Local model | the same from any OpenAI-compatible server you run (llama.cpp, vLLM, Ollama) | no |
| Sources only | the most relevant passages, cited, in order | no |

Settings for the two model modes go in `.env` (see `.env.example`); a mode without settings is disabled. Every
question, the passages used and the answer are appended to the case's `audit.jsonl`.

## Chat

Each chat keeps its own Reactor history, so follow-ups work ("and his brother?"); a follow-up is also searched together
with the previous question. *New chat* starts a fresh history. A period named in a message narrows the search to
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
