"""Finding the passages of a case that answer a question, for the whole case or for one person.

The approach is the one measured on the Memory Retrieval Challenge: BM25 and a sentence embedder over chunks of a few
sentences, rankings fused by reciprocal rank fusion, plus each ranking's own best hits (a passage sharing no word
with the question, "job title" vs "promoted to Director", is still found by the embedder).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from case_assistant.ingest import Case, Person, Unit

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("""
a an the and or but if of to in on at for with from by about as into over after before is are was were be been being
i me my mine myself we us our you your he him his she her it its they them their what which who whom whose when where
why how do does did doing have has had having can could should would will shall may might must this that these those
there here than then so not no yes just also any some all very too more most much many such own same other s t don
""".split())
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
CHUNK_CHARS = 700
RRF_K = 60
DATE_WORDS = re.compile(r"\b(\d{1,4}(st|nd|rd|th)?|between|and|from|to|on|in|after|before|since|until|during|"
                        r"january|february|march|april|may|june|july|august|september|october|november|december|"
                        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", re.I)
# words that only say "tell me what happened": a question made of them and dates lists its window in order
GENERIC = frozenset("happened happen happens going anything everything events event occurred took place tell "
                    "show list timeline overview summary summarise summarize".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    pieces, current = [], ""
    for sentence in SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / self.n if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for term in set(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out


@dataclass
class Passage:
    unit: Unit
    text: str


@dataclass
class Evidence:
    label: str  # S1, S2...
    doc_id: str
    title: str
    date: str
    line: int
    speaker: str
    text: str
    event_date: str | None = None  # ISO
    event_time: str | None = None
    date_source: str = "document"


def alias_pattern(person: Person) -> re.Pattern:
    """A person's name and aliases as whole words (single-word aliases case-sensitively: "Price" is a name, "price"
    is not)."""

    parts = [re.escape(n) if " " in n else rf"(?-i:{re.escape(n)})" for n in person.names]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.I)


class CaseIndex:
    def __init__(self, case: Case, embedder=None):
        self.case, self.embedder = case, embedder
        self.passages: list[Passage] = []
        self.titles = {d.id: d.title for d in case.documents}
        for doc in case.documents:
            for unit in doc.units:
                self.passages += [Passage(unit, piece) for piece in chunk(unit.text)]
        # the document's title goes with every passage ("CCTV review log" says what the passage is)
        self.bm25 = BM25([tokens(f"{self.titles[p.unit.doc_id]} {p.unit.speaker_label} {p.text}") for p in self.passages])
        self.vectors = embedder.docs([self.labelled(p) for p in self.passages]) if embedder and self.passages else None
        self.links = self.link_people()

    def labelled(self, p: Passage) -> str:
        speaker = f"{p.unit.speaker_label}: " if p.unit.speaker_label else ""
        return f"[{self.titles[p.unit.doc_id]}] {speaker}{p.text}"

    def link_people(self) -> dict[str, dict[str, set[str]]]:
        """person id -> {"said": unit keys they speak or wrote, "mentioned": unit keys naming them}."""

        links = {p.id: {"said": set(), "mentioned": set()} for p in self.case.people}
        patterns = {p.id: alias_pattern(p) for p in self.case.people}
        for doc in self.case.documents:
            for unit in doc.units:
                if unit.speaker in links:
                    links[unit.speaker]["said"].add(unit.key)
                for pid, pattern in patterns.items():
                    if pattern.search(unit.text):
                        links[pid]["mentioned"].add(unit.key)
        return links

    def person_units(self, person_id: str) -> set[str]:
        link = self.links.get(person_id, {})
        return link.get("said", set()) | link.get("mentioned", set())

    def rankings(self, query: str) -> list[list[int]]:
        bm25 = self.bm25.scores(tokens(query))
        out = [[i for i in sorted(range(len(bm25)), key=bm25.__getitem__, reverse=True) if bm25[i] > 0]]
        if self.vectors is not None:
            out.append([int(i) for i in np.argsort(-(self.vectors @ self.embedder.query(query)))])
        return out

    @staticmethod
    def dates_inside(p: Passage, start: date | None, end: date | None) -> list[date]:
        return [d for d in p.unit.event_dates if (start is None or d >= start) and (end is None or d <= end)]

    def in_window(self, p: Passage, start: date | None, end: date | None) -> bool:
        return bool(self.dates_inside(p, start, end))

    def event_key(self, p: Passage, start: date | None = None, end: date | None = None) -> tuple:
        """Ordering by when it happened: within a window, the passage's first date inside it."""

        inside = self.dates_inside(p, start, end) if (start or end) else []
        day = inside[0] if inside else (p.unit.event_date or date.max)
        time = p.unit.sort_time if day == p.unit.event_date else "99:99"
        return (day, time, p.unit.doc_id, p.unit.line)

    def search(self, question: str, person: Person | None = None, budget: int = 8000, max_passages: int = 14,
               start: date | None = None, end: date | None = None) -> list[Evidence]:
        """The best passages for ``question`` within ``budget`` characters. With a ``person``, only passages they
        said or that name them; with a date window, only passages about events inside it, in event order."""

        query = f"{question} ({person.name})" if person else question
        allowed = self.person_units(person.id) if person else None
        windowed = start is not None or end is not None
        keep = [i for i, p in enumerate(self.passages)
                if (allowed is None or p.unit.key in allowed) and (not windowed or self.in_window(p, start, end))]
        if not keep:
            return []
        content = [t for t in tokens(DATE_WORDS.sub(" ", question)) if t not in GENERIC]
        listing = windowed and not content  # "what happened between 1 and 31 August?": the window, in order
        if listing:
            pool = sorted(keep, key=lambda i: self.event_key(self.passages[i], start, end))
        else:
            kept = set(keep)
            rankings = [[i for i in r if i in kept] for r in self.rankings(query)]
            fused: dict[int, float] = defaultdict(float)
            for ranking in rankings:
                for rank, i in enumerate(ranking):
                    fused[i] += 1 / (RRF_K + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)
            # the fused best first, then each ranking's own best hits, then the rest of the fused order
            pool = order[:max_passages] + [i for r in rankings for i in r[:max_passages // 2]] + order[max_passages:]
        picked, used, limit = [], 0, max_passages * 2 if listing else max_passages
        for i in dict.fromkeys(pool):
            p = self.passages[i]
            if used + len(p.text) > budget:
                continue
            picked.append(p)
            used += len(p.text)
            if len(picked) >= limit:
                break
        docs = {d.id: (d.date, d.title) for d in self.case.documents}
        picked.sort(key=(lambda p: self.event_key(p, start, end)) if windowed
                    else (lambda p: (docs[p.unit.doc_id][0], p.unit.doc_id, p.unit.line)))
        shown = {id(p): (self.event_key(p, start, end)[0] if windowed else p.unit.event_date) for p in picked}
        return [Evidence(label=f"S{n}", doc_id=p.unit.doc_id, title=docs[p.unit.doc_id][1],
                         date=docs[p.unit.doc_id][0], line=p.unit.line, speaker=p.unit.speaker_label, text=p.text,
                         event_date=shown[id(p)].isoformat() if shown[id(p)] not in (None, date.max) else None,
                         event_time=p.unit.event_time, date_source=p.unit.date_source)
                for n, p in enumerate(picked, 1)]
