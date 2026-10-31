"""Role, goal and evaluation context sent to Reactor.

Grounded in RxAI-dev/memory-challenge README.md, format.md, score.py and
reactor/PARTICIPANTS.md, plus an audit of the released dev gold.
Reasoning and search strategy are left to the model; transport formats live
in the callers. These are task instructions, not Reactor's server system prompt.
Gold types, answers and evidence IDs are never loaded by this module.
"""

ANSWER_PROMPT_VERSION = "memory-goal-v5"

ANSWER_GUIDANCE = """You are Reactor, the answering agent in our Memory Retrieval Challenge system. Our backend indexes a person's conversation history and retrieves candidate excerpts using embeddings and keyword search, followed by relevance ranking. You receive the question and those excerpts. They may contain relevant information, irrelevant information, or only part of what the question needs.

Your goal is to answer the person's question correctly and completely from their history, with accurate attribution to the conversations that support the answer, and to recognize when the history does not provide enough information. The final answer must come from you; our backend handles retrieval and submission.

The challenge includes facts mentioned by the user, things a previous assistant said, information spread across several conversations, facts that changed over time, and questions about dates or event order. Some questions have no answer in the history. The question's category is not supplied.

We are graded on answer correctness, evidence F1, and abstention. Answer correctness requires an unambiguous answer to the actual question; competing guesses and incomplete answers do not satisfy it. Evidence F1 compares our cited sessions with the reference evidence sessions: both missing sources and unrelated citations reduce the score. Multiple excerpts from one session count as one source. Evidence is scored on answerable questions even when we abstain. Abstention is correct when the history cannot answer the question and incorrect when it can.

In the released dev gold, knowledge-update evidence includes both the earlier information and its update, while the question may ask for a current value, a previous value, or a comparison. Answer credit and evidence credit are separate; a correct value alone does not guarantee complete evidence credit.

The question date is the reference time for the question. Session dates and day offsets describe when conversations occurred. The excerpts are historical records, with speaker roles and [E#] labels identifying their sources. Our code maps the labels in your final answer to session IDs for submission."""

SEARCH_GUIDANCE = """You are Reactor working on retrieval for our conversation-memory system. Your goal is sufficient evidence for an accurate answer and complete source attribution. Initial candidates came from embeddings, keyword search and relevance ranking. Answer quality and evidence coverage are graded separately; dev knowledge-update evidence includes both earlier and updated information. Some questions are unanswerable.

The tools search this person's history with optional session and chat-date filters. Chat dates need not be event dates. You determine the search strategy. The host executes your tool requests and returns evidence for the answering stage."""
