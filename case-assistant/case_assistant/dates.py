"""When things happened: dates in passages, and the date window a question asks about.

A passage's event date is the first date its text names ("On Saturday 14 March", "on 2025-08-09", "[14 Mar 20:12]",
"last night" in a report written on the 15th). Without one, it inherits the previous dated passage of the same
document ("I saw a white van..." still describes the 14th), and failing that the date the document was written.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {name.lower(): n for n, name in enumerate(calendar.month_name) if name}
MONTHS |= {name.lower(): n for n, name in enumerate(calendar.month_abbr) if name}
MONTHS["sept"] = 9
MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
DATE = re.compile(
    rf"(?P<iso>\b(?P<iy>\d{{4}})-(?P<im>\d{{2}})-(?P<id>\d{{2}})\b)"
    rf"|(?P<dm>\b(?P<dmd>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<dmm>{MONTH})\.?(?:,?\s+(?P<dmy>\d{{4}}))?\b)"
    rf"|(?P<md>\b(?P<mdm>{MONTH})\.?\s+(?P<mdd>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<mdy>\d{{4}}))?\b)",
    re.I)
MONTH_ONLY = re.compile(rf"\b(?:in|during|throughout)\s+(?P<m>{MONTH})(?:\s+(?P<y>\d{{4}}))?\b", re.I)
RELATIVE = re.compile(r"\b(last night|yesterday|the previous (?:night|evening|day)|this morning|today|tonight)\b", re.I)
TIME = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")


def parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def _resolve(m: re.Match, ref: date) -> date | None:
    g = m.groupdict()
    try:
        if g["iso"]:
            return date(int(g["iy"]), int(g["im"]), int(g["id"]))
        month = MONTHS[(g["dmm"] or g["mdm"]).lower().rstrip(".")]
        day = int(g["dmd"] or g["mdd"])
        year = g["dmy"] or g["mdy"]
        if year:
            return date(int(year), month, day)
        guess = date(ref.year, month, day)
        return guess.replace(year=ref.year - 1) if (guess - ref).days > 60 else guess  # "14 March", said in March
    except (ValueError, KeyError):
        return None


def named_dates(text: str, ref: date) -> tuple[list[date], list[date]]:
    """``(absolute, relative)``: dates written out ("14 March", "2025-08-09"), and ones relative to ``ref``
    ("last night", "the previous night")."""

    absolute = [d for m in DATE.finditer(text) if (d := _resolve(m, ref))]
    relative = [ref if m.group(1).lower() in ("this morning", "today", "tonight") else ref - timedelta(days=1)
                for m in RELATIVE.finditer(text)]
    return absolute, relative


def dates_in(text: str, ref: date) -> list[date]:
    absolute, relative = named_dates(text, ref)
    return absolute + relative


def time_in(text: str) -> str | None:
    m = TIME.search(text)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def window(question: str, ref: date) -> tuple[date | None, date | None, str] | None:
    """The date window a question names: ``(from, to, phrase)``, either end open; None when it names none.

    "between 1 and 31 August 2025", "from 2025-08-01 to 2025-08-31", "on 14 March", "after 10 March",
    "before 9 August 2025", "in August 2025"."""

    q = question
    # "between 1 and 31 August 2025": the first day borrows the second's month and year
    m = re.search(rf"\bbetween\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTH}(?:\s+(\d{{4}}))?",
                  q, re.I)
    if m:
        month, year = MONTHS[m.group(3).lower()], int(m.group(4)) if m.group(4) else ref.year
        try:
            return date(year, month, int(m.group(1))), date(year, month, int(m.group(2))), m.group(0)
        except ValueError:
            pass
    found = [(mm, d) for mm in DATE.finditer(q) if (d := _resolve(mm, ref))]
    years = {d.year for _, d in found}
    if len(found) >= 2:
        a, b = sorted(d for _, d in found[:2])
        if len(years) == 2 and not any(x.group("iy") or x.group("dmy") or x.group("mdy") for x, _ in found[:1]):
            a = a.replace(year=b.year)  # "from 14 March to 2 April 2026"
        return min(a, b), max(a, b), q[found[0][0].start():found[1][0].end()]
    if len(found) == 1:
        mm, d = found[0]
        before = q[:mm.start()].lower()
        if re.search(r"\b(after|since|from)\s*$", before):
            return d, None, q[mm.start() - len(before.split()[-1]) - 1:mm.end()].strip()
        if re.search(r"\b(before|until|till|by)\s*$", before):
            return None, d, q[mm.start() - len(before.split()[-1]) - 1:mm.end()].strip()
        return d, d, mm.group(0)
    m = MONTH_ONLY.search(q)
    if m:
        month = MONTHS[m.group("m").lower()]
        year = int(m.group("y")) if m.group("y") else ref.year
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]), m.group(0)
    return None


def asks_whether(question: str) -> bool:
    """A yes/no question ("Did...", "Was...", "Has...", "Is there...")."""

    return bool(re.match(r"\s*(did|was|were|is|are|has|have|had|does|do|could|can)\b", question, re.I))
