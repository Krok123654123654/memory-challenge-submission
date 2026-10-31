"""Loading a case: its people (people.json) and its documents (documents/*.txt|md|pdf|docx).

A text document may start with a header of ``key: value`` lines ended by ``---``::

    title: Witness statement of Anna Kowalski (first)
    kind: statement
    date: 2026-03-15
    author: anna-kowalski
    ---
    I live at 12 Marlow Lane...

In a transcript (``kind: interview``, ``messages``...), a line starting with a person's alias and a colon is that
person speaking (``DS Brennan: Where were you?``). Every unit keeps its line number, so answers can cite it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from case_assistant.dates import named_dates, parse_iso, time_in


AUTHORED = {"statement", "email", "report", "letter", "note", "document"}  # kinds written in their author's voice


@dataclass
class Person:
    id: str
    name: str
    role: str = ""
    aliases: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Unit:
    """A paragraph of a document, or one speaker's turn in a transcript."""

    doc_id: str
    index: int
    text: str
    line: int
    speaker: str | None = None  # a person id
    speaker_label: str = ""  # as written ("DS Brennan")
    event_date: date | None = None  # when what it describes happened (see dates.py): the first date it names
    event_dates: list[date] = field(default_factory=list)  # every date it names (or the one it inherits)
    event_time: str | None = None  # "22:15", when the text gives a time
    date_source: str = "document"  # "text": named in it; "inherited": from the passage before; "document"
    sort_time: str = "99:99"  # its own time, else the last time before it on the same day: for ordering only

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.index}"


@dataclass
class Document:
    id: str
    title: str
    kind: str
    date: str
    author: str | None
    path: Path
    units: list[Unit] = field(default_factory=list)


@dataclass
class Case:
    name: str
    root: Path
    people: list[Person]
    documents: list[Document]
    meta: dict = field(default_factory=dict)  # case.json: title, type, reference, summary

    @property
    def reference_date(self) -> date:
        """The latest document date: what "14 March" (no year) in a question is read against."""

        dates = [d for d in (parse_iso(doc.date) for doc in self.documents) if d]
        return max(dates) if dates else date.today()

    def person(self, person_id: str) -> Person | None:
        return next((p for p in self.people if p.id == person_id), None)

    def document(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.id == doc_id), None)


def load_people(path: Path) -> list[Person]:
    if not path.exists():
        return []
    return [Person(**p) for p in json.loads(path.read_text(encoding="utf-8"))]


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


def split_header(text: str) -> tuple[dict[str, str], str, int]:
    """``(header, body, line number where the body starts)``."""

    lines = text.splitlines()
    if "---" in [line.strip() for line in lines[:12]]:
        end = [line.strip() for line in lines].index("---")
        header = {}
        for line in lines[:end]:
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip().lower()] = value.strip()
        return header, "\n".join(lines[end + 1:]), end + 2
    return {}, text, 1


def speaker_pattern(people: list[Person]) -> tuple[re.Pattern, dict[str, str]]:
    """A pattern for "Alias:" at the start of a line, and alias -> person id."""

    by_alias = {name.lower(): p.id for p in people for name in p.names}
    names = sorted(by_alias, key=len, reverse=True)
    pattern = re.compile(r"^\s*(" + "|".join(re.escape(n) for n in names) + r")\s*:\s*(.*)$", re.I) if names else None
    return pattern, by_alias


def load_document(path: Path, people: list[Person]) -> Document:
    header, body, first_line = split_header(read_text(path))
    doc = Document(id=path.stem, title=header.get("title", path.stem), kind=header.get("kind", "document"),
                   date=header.get("date", ""), author=header.get("author") or None, path=path)
    pattern, by_alias = speaker_pattern(people)
    current: Unit | None = None
    for offset, line in enumerate(body.splitlines()):
        number = first_line + offset
        match = pattern.match(line) if pattern else None
        if match and match.group(2).strip():  # a new speaker's turn
            current = Unit(doc.id, len(doc.units), match.group(2).strip(), number,
                           speaker=by_alias[match.group(1).lower()], speaker_label=match.group(1))
            doc.units.append(current)
        elif not line.strip():
            current = None  # a blank line ends a paragraph
        elif current is not None:
            current.text += " " + line.strip()
        else:  # a new paragraph: in a statement, email or report its author speaks; in a transcript, nobody
            speaker = doc.author if doc.kind in AUTHORED else None
            current = Unit(doc.id, len(doc.units), line.strip(), number, speaker=speaker,
                           speaker_label=next((p.name for p in people if p.id == speaker), "") if speaker else "")
            doc.units.append(current)
    date_units(doc)
    return doc


TRANSCRIPTS = {"interview", "messages"}


def date_units(doc: Document) -> None:
    """Each unit's event date. Named in its text: that date. Otherwise, in a narrative (statement, report, email),
    the last date written out before it ("On 14 March I..." dates the paragraphs that follow; a relative "the
    previous night" dates only its own paragraph). An interview keeps talking about the date it raised, so its
    turns do the same; lines nobody speaks ("Interview concluded") take the date the transcript was made."""

    written = parse_iso(doc.date) or date.today()
    last_written_out: date | None = None
    for unit in doc.units:
        absolute, relative = named_dates(unit.text, written)
        unit.event_time = time_in(unit.text)
        if absolute or relative:
            unit.event_date, unit.date_source = (absolute or relative)[0], "text"
            unit.event_dates = sorted(set(absolute + relative))
            if absolute:
                last_written_out = absolute[0]
        elif doc.kind in TRANSCRIPTS and not unit.speaker:  # "Interview concluded 10:41."
            unit.event_date, unit.date_source = written, "document"
        elif last_written_out is not None:
            unit.event_date, unit.date_source = last_written_out, "inherited"
        else:
            unit.event_date, unit.date_source = written, "document"
    for unit in doc.units:
        if not unit.event_dates and unit.event_date:
            unit.event_dates = [unit.event_date]
    clock, day = "99:99", None
    for unit in doc.units:
        if unit.event_date != day:
            clock, day = "99:99", unit.event_date
        clock = unit.event_time or clock
        unit.sort_time = clock


def load_case(root: Path | str) -> Case:
    root = Path(root)
    people = load_people(root / "people.json")
    files = sorted(p for p in (root / "documents").iterdir() if p.suffix.lower() in (".txt", ".md", ".pdf", ".docx"))
    documents = [load_document(p, people) for p in files]
    documents.sort(key=lambda d: (d.date, d.id))
    meta = json.loads((root / "case.json").read_text(encoding="utf-8")) if (root / "case.json").exists() else {}
    return Case(name=root.name, root=root, people=people, documents=documents, meta=meta)
