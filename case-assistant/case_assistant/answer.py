"""Answers from the retrieved passages, with citations, through a choice of model.

- ``RetrievalOnly``: no model at all; shows the passages found (instant, nothing leaves the machine).
- ``OpenAICompatible``: OpenRouter, or any OpenAI-compatible server you run yourself (llama.cpp, vLLM, Ollama...).
- ``Reactor``: the hackathon's hosted agent, through a small client of our own; for synthetic demo data only.

Every question, the passages used and the answer go to the case's ``audit.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from case_assistant.search import Evidence

NOT_IN_FILE = "NOT IN FILE"
FINAL = re.compile(r"^[ \t]*#{1,3}[ \t]*final[ _-]?answer[ \t]*\[[ \t]*ref[ \t]*:[ \t]*([0-9a-f]{8})[ \t]*\][ \t]*:?[ \t]*",
                   re.I | re.M)

INSTRUCTIONS = """You are assisting a legal team (prosecution or defence) reviewing a case file. Answer the question \
using only the numbered sources below.
- Cite every source you rely on by its label, e.g. [S2], including on the final answer line.
- Keep straight who said what: what one person says about another is not that other person's own statement, and a \
plan is not something that happened.
- If sources conflict (for example a witness changed their account), say so and cite each version.
- If the sources do not contain the answer, the answer is {not_in_file}.
- State what the sources say; do not speculate about guilt or fill gaps."""


@dataclass
class Answer:
    question: str
    person: str | None
    evidence: list[Evidence]
    text: str | None = None  # None: retrieval only
    cited: list[str] = field(default_factory=list)  # S labels
    backend: str = "retrieval only"
    seconds: float = 0.0
    error: str | None = None
    window: list[str | None] | None = None  # [from, to] ISO, either end open
    window_phrase: str = ""  # what the window was read from ("between 1 and 31 August")
    verdict: str | None = None  # "yes" / "no" / "not in file" for a did-it-happen question


def build_prompt(question: str, evidence: list[Evidence], ref: str, person_name: str | None = None,
                 window: list[str | None] | None = None, whether: bool = False) -> str:
    focus = f"\nThe question is about {person_name}; the sources are the passages they said or that name them." \
        if person_name else ""
    if window:
        start, end = window
        span = f"from {start} to {end}" if start and end else f"from {start} on" if start else f"up to {end}"
        focus += (f"\nThe question is about events {span}; the sources are passages about that period, in the order "
                  "the events happened. Describe what happened in that order, with dates and times.")
    if whether:
        focus += (f"\nThe question asks whether something happened. Start the final answer with one word: Yes or No "
                  f"when the sources establish it; Disputed when sources contradict each other (give each side and "
                  f"who says it, with its citation); {NOT_IN_FILE} when the sources are silent.")
    sources = "\n".join(f"[{e.label}] ({e.title}, written {e.date}, line {e.line}"
                        + (f", about {e.event_date}" + (f" {e.event_time}" if e.event_time else "")
                           if e.event_date and e.date_source != "document" else "")
                        + (f", {e.speaker} speaking" if e.speaker else "") + f") {e.text}" for e in evidence)
    return (f"{INSTRUCTIONS.format(not_in_file=NOT_IN_FILE)}{focus}\n\nSources:\n{sources}\n\nQuestion: {question}\n\n"
            f"Think it through, then write the answer on a line of its own starting with: # FINAL ANSWER [ref:{ref}]: "
            f"followed by the answer itself in one to four plain sentences, with [S#] labels after the facts they "
            f"support.")


def verdict_of(text: str | None) -> str | None:
    if not text:
        return None
    head = text.strip().lower()
    if head.startswith(NOT_IN_FILE.lower()):
        return "not in file"
    for word in ("yes", "no", "disputed"):
        if re.match(rf"{word}\b", head):
            return word
    return None


def final_answer(reply: str, ref: str) -> str:
    marks = [m for m in FINAL.finditer(reply) if m.group(1).lower() == ref]
    return reply[marks[-1].end():].strip() if marks else reply.strip()


def cited_labels(text: str) -> list[str]:
    return sorted({f"S{n}" for n in re.findall(r"\bS(\d+)\b", text)}, key=lambda s: int(s[1:]))


class RetrievalOnly:
    name = "retrieval only"

    def ask(self, prompt: str, ref: str) -> str | None:
        return None


class OpenAICompatible:
    """Any OpenAI-compatible chat API: OpenRouter, or a server you run yourself (llama.cpp, vLLM, Ollama). The API
    keeps no state, so this keeps the chat's earlier questions and answers and sends them along."""

    SYSTEM = "You assist a legal team reviewing a case file. Answer only from the sources given with each question."

    def __init__(self, base_url: str, model: str, api_key: str | None = None, name: str = "model",
                 timeout: float = 300, keep_turns: int = 6):
        self.url, self.model, self.key, self.name = base_url.rstrip("/"), model, api_key, name
        self.timeout, self.keep_turns = timeout, keep_turns
        self.turns: list[tuple[str, str]] = []  # (question, final answer) of this chat

    def ask(self, prompt: str, ref: str) -> str:
        messages = [{"role": "system", "content": self.SYSTEM}]
        for question, reply in self.turns[-self.keep_turns:]:
            messages += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json", "X-Title": "Case assistant"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = json.dumps({"model": self.model, "temperature": 0, "messages": messages}).encode()
        req = urllib.request.Request(f"{self.url}/chat/completions", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code} from {self.name}: {detail}") from exc

    def remember(self, question: str, reply: str) -> None:
        self.turns.append((question, reply))


def LocalLLM(url: str, model: str) -> OpenAICompatible:  # noqa: N802  a server you run: /v1 under its address
    return OpenAICompatible(f"{url.rstrip('/')}/v1", model, name="local model")


class Reactor:
    """The hackathon's Reactor agent. One Reactor history per case session; messages one at a time."""

    name = "Reactor"
    RETRYABLE = ("busy", "previous_pending")

    def __init__(self, base_url: str, agent_id: str, access_key: str, timeout: float = 300):
        if not base_url.startswith("https://"):
            raise ValueError("Reactor base URL must start with https://")
        self.base, self.agent, self.key, self.timeout = base_url.rstrip("/"), agent_id, access_key, timeout
        self.history: str | None = None

    def new_history(self) -> str:
        req = urllib.request.Request(f"{self.base}/agents/{self.agent}/history", method="POST", data=b"",
                                     headers={"Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.history = json.loads(resp.read())["history_id"]
        return self.history

    async def _ask(self, text: str, ref: str) -> str:
        import websockets

        url = f"wss://{self.base[len('https://'):]}/agents/{self.agent}/ws/chat?history={quote(self.history, safe='')}"
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {self.key}"}) as ws:
            await ws.send(json.dumps({"type": "message", "text": text}))
            ack = None
            async for raw in ws:
                frame = json.loads(raw)
                kind = frame.get("type")
                if kind == "error":
                    raise RuntimeError(f"{frame.get('code')}: {frame.get('text')}", frame.get("code"))
                if kind == "ack" and ack is None and frame.get("text") == text:
                    ack = frame.get("cycle", 0)
                elif kind == "reply" and ack is not None and isinstance(frame.get("cycle"), int) \
                        and frame["cycle"] > ack and any(m.group(1).lower() == ref for m in FINAL.finditer(
                            frame.get("text") or "")):
                    return frame["text"]
        raise RuntimeError("the chat closed before an answer")

    def ask(self, prompt: str, ref: str) -> str:
        if self.history is None:
            self.new_history()
        deadline, delay = time.monotonic() + self.timeout, 1.0
        while True:
            try:
                return asyncio.run(asyncio.wait_for(self._ask(prompt, ref), max(deadline - time.monotonic(), 1)))
            except RuntimeError as exc:
                code = exc.args[1] if len(exc.args) > 1 else None
                if code in self.RETRYABLE and deadline - time.monotonic() > delay:
                    time.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
                raise


def answer(backend, question: str, evidence: list[Evidence], person_name: str | None = None,
           audit: Path | None = None, window: list[str | None] | None = None, window_phrase: str = "",
           whether: bool = False) -> Answer:
    result = Answer(question=question, person=person_name, evidence=evidence, backend=backend.name,
                    window=window, window_phrase=window_phrase)
    if evidence and not isinstance(backend, RetrievalOnly):
        ref = secrets.token_hex(4)
        started = time.monotonic()
        try:
            reply = backend.ask(build_prompt(question, evidence, ref, person_name, window, whether), ref)
            result.text = final_answer(reply, ref)
            if hasattr(backend, "remember"):  # a stateless API: keep the turn for the next question
                backend.remember(question, result.text)
            result.verdict = verdict_of(result.text) if whether else None
            result.cited = cited_labels(result.text) or cited_labels(reply)
        except Exception as exc:  # shown in the app, and logged
            result.error = str(exc)
        result.seconds = round(time.monotonic() - started, 1)
    elif not evidence and not isinstance(backend, RetrievalOnly):
        result.text = f"{NOT_IN_FILE} (no passage matched)"
        result.verdict = "not in file" if whether else None
    if audit:
        with audit.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **asdict(result)},
                               default=str) + "\n")
    return result
