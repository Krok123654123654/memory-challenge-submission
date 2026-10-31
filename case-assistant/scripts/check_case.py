"""Checks retrieval on a case's questions.jsonl: are the expected documents among the passages found?

    python scripts/check_case.py cases/marlow-street            # BM25 + bge-small
    python scripts/check_case.py cases/marlow-street --no-embed # BM25 only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from case_assistant.dates import window  # noqa: E402
from case_assistant.ingest import load_case  # noqa: E402
from case_assistant.search import CaseIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    case = load_case(args.case)
    embedder = None
    if not args.no_embed:
        from case_assistant.embedder import Embedder
        embedder = Embedder()
    index = CaseIndex(case, embedder)
    print(f"{len(case.documents)} documents, {sum(len(d.units) for d in case.documents)} units, "
          f"{len(index.passages)} passages, {len(case.people)} people")
    found = total = 0
    for line in (args.case / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        person = case.person(q["person"]) if q.get("person") else None
        when = window(q["question"], case.reference_date)  # as the server does
        evidence = index.search(q["question"], person=person, start=when[0] if when else None,
                                end=when[1] if when else None)
        docs = {e.doc_id for e in evidence}
        hit = all(d in docs for d in q["docs"])
        found += hit and bool(q["docs"]); total += bool(q["docs"])
        mark = "-" if not q["docs"] else ("ok" if hit else "MISS")
        span = f" window {when[0]}..{when[1]}" if when else ""
        print(f"[{mark:4}] {q['question'][:62]:62} {len(evidence):2} passages{span}"
              + (f"  missing {sorted(set(q['docs']) - docs)}" if q["docs"] and not hit else ""))
    print(f"\nall expected documents found for {found}/{total} answerable questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
