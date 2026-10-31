"""Local history search shared by fixed retrieval and the MCP tool server.

No gold labels, Reactor credentials, or external inference are used here.
Date filters apply to chat dates, not inferred event dates.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from retrieval import date
from model_config import RETRIEVAL_PROFILE, RERANKER_PIN, require_current_profile
from evidence_packing import MAX_PASSAGES, annotated, calendar_offset, pack_fragments

KIT = Path(__file__).resolve().parent


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def date_bounds(question, date_from=None, date_to=None):
    def parse(value):
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date filters must be YYYY-MM-DD calendar dates")
        return datetime.strptime(value, "%Y-%m-%d").date()
    lo = parse(date_from) if date_from is not None else None
    hi = parse(date_to) if date_to is not None else date(question["question_date"]).date()
    hi = min(hi, date(question["question_date"]).date())
    if lo is not None and lo > hi:
        raise ValueError("date_from must not be after date_to or the question date")
    return lo, hi


def allowed_sessions(question, history, date_from=None, date_to=None, session_ids=None):
    lo, hi = date_bounds(question, date_from, date_to)
    valid_ids = {s["session_id"] for s in history["sessions"]}
    if session_ids is not None:
        if not isinstance(session_ids, list) or not all(isinstance(s, str) for s in session_ids):
            raise ValueError("session_ids must be a list of chat IDs")
        if not set(session_ids) <= valid_ids:
            raise ValueError("A requested chat does not belong to this user's history")
    return {s["session_id"] for s in history["sessions"]
            if date(s["date"]) <= date(question["question_date"])
            and (lo is None or date(s["date"]).date() >= lo)
            and date(s["date"]).date() <= hi
            and (session_ids is None or s["session_id"] in session_ids)}


class SearchBackend:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        require_current_profile(self.root)
        self.questions = {q["question_id"]: q for q in read_jsonl(self.root / "questions.jsonl")}
        self.histories = {h["history_id"]: h for h in read_jsonl(self.root / "histories.jsonl")}
        self.indexes = {}
        self.embedder = self.reranker = None

    def models(self):
        if self.embedder is not None:
            return
        from openprovence_backend import OpenProvenceRetriever
        self.engine = OpenProvenceRetriever(self.root / "cache")
        self.embedder = self.engine.embedding.model
        self.reranker = self.engine.pruner

    def index(self, question):
        self.models()
        hid = question["history_id"]
        if hid not in self.engine.indexes:
            self.engine.indexes[hid] = self.engine.embedding.index(self.histories[hid])
        return self.engine.indexes[hid]

    def rank(self, query, fragments):
        self.models()
        return self.reranker.score(query, fragments)[0]

    def search_history(self, question_id, query, date_from=None, date_to=None, session_ids=None,
                       limit=MAX_PASSAGES, budget=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must be 1 to 500 characters")
        if type(limit) is not int or not 1 <= limit <= MAX_PASSAGES:
            raise ValueError("limit must be an integer from 1 to 12")
        q = self.questions[question_id]
        allowed = allowed_sessions(q, self.histories[q["history_id"]], date_from, date_to, session_ids)
        if not allowed:
            return {**pack_fragments(q, [], limit=limit, evidence_budget=budget),
                    "matching_chats": 0, "seconds": 0, "retrieval_profile": RETRIEVAL_PROFILE,
                    "date_filter_semantics": "inclusive chat dates, not event dates"}
        self.models()
        return {**self.engine.search(q, self.histories[q["history_id"]], query=query,
                                    allowed_ids=allowed, limit=limit, budget=budget),
                "matching_chats": len(allowed)}

    def list_chats(self, question_id, date_from=None, date_to=None, offset=0, limit=25):
        q = self.questions[question_id]
        h = self.histories[q["history_id"]]
        allowed = allowed_sessions(q, h, date_from, date_to)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 25:
            raise ValueError("offset must be nonnegative; limit must be 1 to 25")
        sessions = sorted([s for s in h["sessions"] if s["session_id"] in allowed], key=lambda s: date(s["date"]))
        return {"chats": [{"session_id": s["session_id"], "date": s["date"],
                 "days_before_question": calendar_offset(q["question_date"],s["date"]),
                 "turns": len(s["turns"])} for s in sessions[offset:offset+limit]],
                "next_offset": offset+limit if offset+limit < len(sessions) else None,
                "matching_chats": len(sessions), "date_filter_semantics": "inclusive chat dates, not event dates"}


DATE_FIELDS = {"date_from": {"type": "string", "description": "Inclusive chat date YYYY-MM-DD; omit for no lower bound"},
               "date_to": {"type": "string", "description": "Inclusive chat date YYYY-MM-DD; never beyond question date"}}
TOOLS = [
    {"name": "search_history", "description": "Search this user's chats by meaning and keywords, rerank with BGE, return original excerpts and dates. Date filters refer to chat dates, not event dates.",
     "inputSchema": {"type": "object", "properties": {
         "question_id": {"type": "string", "description": "Bound by the host; cannot be changed by the model"},
         "query": {"type": "string", "minLength": 1, "maxLength": 500}, **DATE_FIELDS,
         "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional chat IDs to restrict search to"},
         "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PASSAGES, "default": 4}},
         "required": ["question_id", "query"], "additionalProperties": False}},
    {"name": "list_chats", "description": "List this user's chat IDs and dates, optionally within a chat-date range. Use those IDs to restrict search_history.",
     "inputSchema": {"type": "object", "properties": {
         "question_id": {"type": "string"}, **DATE_FIELDS,
         "offset": {"type": "integer", "minimum": 0, "default": 0},
         "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 25}},
         "required": ["question_id"], "additionalProperties": False}}
]
