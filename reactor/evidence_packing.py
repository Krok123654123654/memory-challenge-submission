"""Pack ranked evidence against the exact formatted Reactor message size."""
from answer_guidance import ANSWER_GUIDANCE
from client import MAX_CHARS
from retrieval import date

MAX_PASSAGES = 12


def calendar_offset(question_date, session_date):
    return (date(question_date).date() - date(session_date).date()).days


def annotated(question, fragment):
    days = calendar_offset(question["question_date"], fragment.date)
    return {"id": fragment.id, "date": fragment.date,
            "text": f"Chat occurred {days} calendar days before the question date. "
                    "This is the chat date, not necessarily the date of events described.\n" + fragment.text}


def evidence_block(evidence):
    return "\n".join(f"[E{i}] (source: {e['id']}, date: {e['date']}) {e['text']}"
                     for i, e in enumerate(evidence, 1))


def answer_message(question, evidence, ref):
    return (f"[ref:{ref}] Question: {question['question']}\n"
        f"Question date: {question['question_date']}.\n{ANSWER_GUIDANCE}\n\nEvidence:\n"
        + evidence_block(evidence)
        + '\n\nReturn JSON: {"action":"answer","answer":"your answer [E1]","abstain":false}. '
        'Citations belong inside the answer string. For abstention, abstain is true and the answer is '
        'empty or INSUFFICIENT_EVIDENCE with any partial-evidence labels.\n'
        f'Write that JSON on one line starting exactly with # FINAL ANSWER [ref:{ref}]:')


def pack_fragments(question, ranked, *, limit=MAX_PASSAGES, one_per_session=False,
                   evidence_budget=None, max_chars=MAX_CHARS):
    if type(limit) is not int or not 1 <= limit <= MAX_PASSAGES:
        raise ValueError(f"limit must be an integer from 1 to {MAX_PASSAGES}")
    if evidence_budget is not None and (type(evidence_budget) is not int or evidence_budget < 0):
        raise ValueError("evidence budget must be a nonnegative integer")
    if len(answer_message(question, [], "00000000")) > max_chars:
        raise ValueError("Question and instructions exceed the message budget")
    selected = []
    evidence = []
    skipped_budget = 0
    for f in ranked:
        if len(selected) == limit:
            break
        if one_per_session and any(f.id == old.id for old in selected):
            continue
        overlap = any(f.id == old.id and f.turn_id == old.turn_id and
            max(0, min(f.end, old.end) - max(f.start, old.start)) /
            max(1, min(f.end - f.start, old.end - old.start)) > .5 for old in selected)
        if overlap or not f.text.strip():
            continue
        ordered = sorted(selected + [f], key=lambda x: (date(x.date), x.id, x.turn_index, x.start))
        proposed = [annotated(question, x) for x in ordered]
        if ((evidence_budget is not None and len(evidence_block(proposed)) > evidence_budget)
                or len(answer_message(question, proposed, "00000000")) > max_chars):
            skipped_budget += 1
            continue
        selected.append(f)
        evidence = proposed
    chars = len(answer_message(question, evidence, "00000000"))
    return {"evidence": evidence, "packed_chars": len(evidence_block(evidence)),
            "message_chars": chars, "remaining_chars": max_chars - chars,
            "passages": len(evidence), "skipped_for_budget": skipped_budget}
