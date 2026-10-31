"""Case assistant: a small web app over a case file (people, documents, cited answers).

    uvicorn server:app --host 127.0.0.1 --port 8000      # then open http://127.0.0.1:8000

An assistant for reviewing a case file, for prosecution or defence. It finds, orders and cites; it does not decide.
It serves only this machine (127.0.0.1): the case files are not meant to leave it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from case_assistant.answer import LocalLLM, OpenAICompatible, Reactor, RetrievalOnly, answer
from case_assistant.dates import asks_whether, parse_iso, window
from case_assistant.ingest import load_case
from case_assistant.search import CaseIndex, tokens

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
STATIC = ROOT / "static"
EMBEDDER = "BAAI/bge-small-en-v1.5"


def load_env() -> None:
    """Settings from .env; read again on every check, so a key added there works without a restart."""

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


load_env()
app = FastAPI(title="Case assistant", docs_url=None, redoc_url=None)
# chat sessions: id -> {"reactor": Reactor or None, "questions": [...]}; one Reactor history per chat
SESSIONS: dict[str, dict] = {}


@lru_cache(maxsize=1)
def embedder():
    from case_assistant.embedder import Embedder

    return Embedder(EMBEDDER)


@lru_cache(maxsize=8)
def get_index(case_name: str, semantic: bool) -> CaseIndex:
    path = (CASES / case_name).resolve()
    if path.parent != CASES.resolve() or not (path / "documents").is_dir():
        raise HTTPException(404, f"no case {case_name!r}")
    return CaseIndex(load_case(path), embedder() if semantic else None)


def backends() -> dict[str, bool]:
    """Each answer backend, and whether it is configured."""

    load_env()
    return {
        "retrieval": True,
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_MODEL")),
        "local": bool(os.environ.get("LOCAL_LLM_URL") and os.environ.get("LOCAL_LLM_MODEL")),
        "reactor": all(os.environ.get(k) for k in ("REACTOR_BASE_URL", "REACTOR_AGENT_ID", "REACTOR_ACCESS_KEY")),
    }


def backend(kind: str, session: dict | None = None):
    if session is None:
        session = {}
    if kind == "openrouter" and backends()["openrouter"]:
        if session.get("openrouter") is None:  # keeps this chat's turns
            session["openrouter"] = OpenAICompatible("https://openrouter.ai/api/v1", os.environ["OPENROUTER_MODEL"],
                                                     os.environ["OPENROUTER_API_KEY"], name=os.environ["OPENROUTER_MODEL"])
        return session["openrouter"]
    if kind == "local" and backends()["local"]:
        if session.get("local") is None:
            session["local"] = LocalLLM(os.environ["LOCAL_LLM_URL"], os.environ["LOCAL_LLM_MODEL"])
        return session["local"]
    if kind == "reactor" and backends()["reactor"]:
        if session.get("reactor") is None:  # a new chat: a new Reactor history on its first message
            session["reactor"] = Reactor(os.environ["REACTOR_BASE_URL"], os.environ["REACTOR_AGENT_ID"],
                                         os.environ["REACTOR_ACCESS_KEY"])
        return session["reactor"]
    if kind != "retrieval":
        raise HTTPException(400, f"backend {kind!r} is not configured in .env")
    return RetrievalOnly()


def span(dates: list[str]) -> str:
    if not dates:
        return "—"
    a, b = date.fromisoformat(min(dates)[:10]), date.fromisoformat(max(dates)[:10])
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}–{b.day} {b:%b %Y}"
    if a.year == b.year:
        return f"{a.day} {a:%b} – {b.day} {b:%b %Y}"
    return f"{a.day} {a:%b %Y} – {b.day} {b:%b %Y}"


@app.get("/api/config")
def config():
    cases = sorted(p.name for p in CASES.iterdir() if (p / "documents").is_dir())
    ready = backends()
    # Reactor is the main mode; OpenRouter and a local model are alternatives. ANSWERS_DEFAULT in .env overrides.
    wanted = os.environ.get("ANSWERS_DEFAULT", "")
    default = wanted if ready.get(wanted) else next(k for k in ("reactor", "openrouter", "local", "retrieval") if ready[k])
    return {"cases": cases, "backends": ready, "default_backend": default,
            "openrouter_model": os.environ.get("OPENROUTER_MODEL", "")}


@app.get("/api/cases/{case_name}")
def case_summary(case_name: str, semantic: bool = True):
    index = get_index(case_name, semantic)
    case = index.case
    people = []
    for p in case.people:
        link = index.links[p.id]
        people.append({**asdict(p), "said_docs": len({k.split("#")[0] for k in link["said"]}),
                       "mentioned_docs": len({k.split("#")[0] for k in link["mentioned"] - link["said"]})})
    documents = [{"id": d.id, "title": d.title, "kind": d.kind, "date": d.date, "author": d.author,
                  "people": [p.id for p in case.people if any(u.key in index.person_units(p.id) for u in d.units)]}
                 for d in case.documents]
    examples = []
    questions = case.root / "questions.jsonl"
    if questions.exists():
        rows = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line.strip()]
        examples = [r["question"] for r in rows if not r.get("person")][:4]
    meta = case.meta
    return {"name": case.name, "title": meta.get("title") or case.name.replace("-", " ").title(),
            "type": meta.get("type", ""), "reference": meta.get("reference", ""), "summary": meta.get("summary", ""),
            "people": people, "documents": documents, "passages": len(index.passages),
            "dates": span([d.date for d in case.documents if d.date]), "examples": examples}


@app.get("/api/cases/{case_name}/documents/{doc_id}")
def document(case_name: str, doc_id: str):
    doc = get_index(case_name, False).case.document(doc_id)
    if not doc:
        raise HTTPException(404, f"no document {doc_id!r}")
    return {"id": doc.id, "title": doc.title, "kind": doc.kind, "date": doc.date, "author": doc.author,
            "units": [{"line": u.line, "speaker": u.speaker, "speaker_label": u.speaker_label, "text": u.text}
                      for u in doc.units]}


@app.get("/api/cases/{case_name}/people/{person_id}")
def person(case_name: str, person_id: str):
    index = get_index(case_name, False)
    p = index.case.person(person_id)
    if not p:
        raise HTTPException(404, f"no person {person_id!r}")
    said, mentioned = index.links[p.id]["said"], index.links[p.id]["mentioned"]

    def grouped(keys: set[str]) -> list[dict]:
        out = []
        for doc in index.case.documents:
            units = [{"line": u.line, "speaker_label": u.speaker_label, "text": u.text}
                     for u in doc.units if u.key in keys]
            if units:
                out.append({"id": doc.id, "title": doc.title, "kind": doc.kind, "date": doc.date, "units": units})
        return out

    return {**asdict(p), "own_words": grouped(said), "mentioned_by_others": grouped(mentioned - said)}


class Message(BaseModel):
    session: str  # a chat: the browser makes the id
    message: str
    person: str | None = None  # ask about one person only
    backend: str = "reactor"
    semantic: bool = True


FOLLOW_UP = re.compile(r"^\s*(and|but|what about|how about|why|when|where|did (he|she|they)|was (he|she|it)|"
                       r"is (he|she|it)|(he|she|they|it|his|her|their|that|this|those)\b)", re.I)


@app.post("/api/cases/{case_name}/chat")
def chat(case_name: str, m: Message):
    """One chat turn: find the passages, ask the model, cite. A follow-up ("and his brother?") searches with the
    previous question too; the model keeps the conversation itself (one Reactor history per chat)."""

    if not m.message.strip():
        raise HTTPException(400, "empty message")
    index = get_index(case_name, m.semantic)
    session = SESSIONS.setdefault(f"{case_name}/{m.session}", {"questions": []})
    person = index.case.person(m.person) if m.person else None
    query = m.message
    if session["questions"] and (FOLLOW_UP.match(m.message) or len(tokens(m.message)) <= 3):
        query = f"{session['questions'][-1]} {m.message}"
    session["questions"].append(m.message)
    start, end, phrase = None, None, ""
    if found := window(m.message, index.case.reference_date):
        start, end, phrase = found
    evidence = index.search(query, person=person, start=start, end=end)
    span_ = [start.isoformat() if start else None, end.isoformat() if end else None] if (start or end) else None
    result = answer(backend(m.backend, session), m.message, evidence, person.name if person else None,
                    audit=index.case.root / "audit.jsonl", window=span_, window_phrase=phrase,
                    whether=asks_whether(m.message))
    return asdict(result)


class Reset(BaseModel):
    session: str


@app.post("/api/cases/{case_name}/chat/reset")
def reset_chat(case_name: str, r: Reset):
    """A new chat: forget the questions, and the next Reactor message starts a new history."""

    SESSIONS.pop(f"{case_name}/{r.session}", None)
    return {"ok": True}


@app.get("/api/cases/{case_name}/timeline")
def timeline(case_name: str, date_from: str | None = None, date_to: str | None = None, person: str | None = None):
    """Every passage about a dated event (dated in its text, or by the passage before it), in event order."""

    index = get_index(case_name, False)
    start, end = parse_iso(date_from or ""), parse_iso(date_to or "")
    allowed = index.person_units(person) if person else None
    titles = {d.id: (d.title, d.kind) for d in index.case.documents}
    events = []  # a passage naming several dates appears once per date in the window
    for d in index.case.documents:
        for u in d.units:
            if u.date_source == "document" or (allowed is not None and u.key not in allowed):
                continue
            for day in u.event_dates:
                if (start is None or day >= start) and (end is None or day <= end):
                    events.append((day, u.sort_time if day == u.event_date else "99:99", u))
    events.sort(key=lambda e: (e[0], e[1], e[2].doc_id, e[2].line))
    return {"events": [{"date": day.isoformat(), "time": u.event_time if day == u.event_date else None, "doc_id": u.doc_id,
                        "title": titles[u.doc_id][0], "kind": titles[u.doc_id][1], "line": u.line,
                        "speaker": u.speaker_label, "text": u.text,
                        "people": [p.id for p in index.case.people if u.key in index.person_units(p.id)]}
                       for day, _, u in events]}


@app.middleware("http")
async def no_stale_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"  # always revalidate: an updated app.js shows at once
    return response


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")
