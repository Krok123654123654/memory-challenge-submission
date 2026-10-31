# Presentation outline

## Problem

Find a few relevant facts in a user's long, noisy conversation archive. Handle near misses, facts from either speaker, updates, calculations, time references and missing information.

## Pipeline

1. Preserve speaker roles, session identity, dates and neighboring-turn context while chunking.
2. Combine BGE-large semantic retrieval with BM25 using reciprocal rank fusion.
3. Rank the top 40 candidates with MIT OpenProvence and prune sentence units at 0.05.
4. Pack up to 12 passages inside a 16,000-character complete message. Include session dates and day offsets.
5. Ask Reactor using a role/goal/evaluation prompt. Keep one Reactor history per user and process questions sequentially.
6. Map final-answer labels to session IDs; retain exact responses and durable checkpoints.

## Evidence so far

- Twelve selected comparison cases matched source review, including two reported failures of the friend's approach. This was not an official test score; seven cases lacked local gold.
- On the five dev cases in that comparison, the local matcher marked three correct and two paraphrases unjudged. Citation F1 across four answerable cases was 0.917.
- Separate 24-turn chess and 32-turn roleplay support conversations both passed local dev scoring with evidence F1 1.0.
- The final 146-question run must finish before reporting a full submission result. Do not present these selected-case results as overall test accuracy.

## Engineering lessons

- A correct session ID does not guarantee that the required sentence survived pruning. One inspected temporal question retained the source session but lost its registration-date statement.
- Multiple excerpts from a conversation preserve facts that a one-excerpt restriction can omit.
- A timed-out client request can still be running remotely. Receive late replies without resets or duplicate submissions.
- Cached embeddings and frozen evidence remove GPU work from the slow Reactor answering loop.

## Limits and next work

Citation selection can still include distractors or omit reference sessions. Pruning can remove needed facts. We have not isolated the causal effect of the prompt, embeddings or packing changes, and official test judging is unavailable locally.
