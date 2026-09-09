# Fix: OpenAI API cost steadily increasing

## Context
The user reports their OpenAI API bill for this RAG chatbot keeps climbing over time (not just with raw traffic volume). Investigation of `image/src/rag_app/core_ai.py` found two background LLM calls — fired on **every single user message** via `asyncio.create_task` (`core_ai.py:579-580`) — whose prompt size grows **unboundedly with usage**, rather than staying flat per message. This is the specific pattern that explains "steadily increasing" cost as opposed to "cost proportional to traffic": the same number of messages costs more today than it did last month, and will cost more next month than today.

## Root causes (confirmed by reading the code directly)

**1. Topic classification re-sends the entire app-wide topic history on every message — unbounded, global, forever-growing.**
`core_ai.py:219-226` (`classify_question_background`):
```python
existing_topics = (
    db.query(models.ChatLog.topic)
    .filter(models.ChatLog.topic != None)
    .distinct()
    .all()
)
```
This queries **every distinct topic ever assigned, across all sessions, all users, since the app launched** — no `LIMIT`, no scoping to recent activity. The full list is then interpolated into `topic_prompt` (`core_ai.py:229-241`) and sent to `gpt-4o-mini` for classification of *every* incoming question. As the institute's chatbot accumulates more distinct topics month over month, this one prompt grows and grows, and it's paid on every message, not just once.

**2. Intent detection re-sends the full session's message history on every message — grows quadratically per long session.**
`core_ai.py:265-356` (`detect_intent_background`), lines 278-289: fetches **all** user messages in the session with no cap, joins them into `user_questions_text`, and sends the whole thing to the LLM (`core_ai.py:321`) to re-derive `price_intent`/`registration_intent`. This repeats on every new message in that session until both flags are already `True` (there's an early-exit at line 274, but only once both intents are already confirmed). For a session of length N, total tokens sent across the session scale ~O(N²) — a customer having a long back-and-forth gets progressively more expensive per turn, and this compounds as average session length grows over time.

**Contributing but secondary**: `GLOBAL_FACTS` (`core_ai.py:40-92`, ~450-500 tokens) is injected into every RAG answer prompt — a flat per-message tax, not a growth driver, but worth trimming if it keeps expanding. Retrieval (`TOP_K_RESULTS=5`, `SIMILARITY_THRESHOLD=1.5`), `MEMORY_WINDOW_SIZE=3` history slicing, and embeddings were all checked and are correctly bounded — not contributing to the trend.

## Fix approach

**Fix 1 — bound the topic list sent to the classifier** (`core_ai.py:219-226`):
Cap the query with a `LIMIT` (e.g. most-recent N distinct topics, or cap the count returned) so the prompt size stays flat regardless of how many topics the app has accumulated historically. Recent topics are what matter for clustering new questions anyway — ancient one-off topics add cost without improving classification quality.

**Fix 2 — bound the intent-detection history window** (`core_ai.py:278-289`):
Instead of fetching *all* user messages in the session, cap to the most recent K messages (e.g. last 10-15) via `.order_by(...desc()).limit(K)` then re-reverse for chronological order. This keeps the per-call prompt size flat instead of growing with session length, while still catching intent signals from "delayed" messages within a reasonable recent window (per the existing design rationale in CLAUDE.md).

Both fixes are small, surgical changes to existing query logic — no new abstractions, no schema changes, no behavior change beyond capping unbounded queries.

## Files to modify
- `image/src/rag_app/core_ai.py` — lines 219-226 (topic query) and 278-289 (intent history query)

## Verification
1. Run the server locally (`uvicorn main:app --reload` per CLAUDE.md).
2. Send several chat messages in one session; confirm topic classification and intent detection still work (check logs for `[CLASSIFY]` output and lead status updates in the dashboard).
3. Manually inspect the generated `topic_prompt` and `intent_prompt` (temporarily print their length) before/after the fix to confirm size no longer scales with total historical topics / full session length.
4. Optionally seed the DB with many `ChatLog` rows across many sessions/topics to simulate an "aged" install and confirm the classification prompt stays bounded.
