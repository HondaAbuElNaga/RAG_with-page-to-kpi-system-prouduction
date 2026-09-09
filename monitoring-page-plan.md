# Add `/monitoring` page (RAG logging stopgap before Grafana/Prometheus)

## Context

The project is hosted on AWS Fargate, and container stdout is already shipped to CloudWatch Logs (group `/ecs/sstli-chatbot`, confirmed in `ecs-task-definition.json:52-57` / `task-definition-v2.json:37-42`, `awslogs` driver). Right now the only RAG-quality signal is a single structured print in `core_ai.py:570-578` (`log_type: "RAG_EVAL"` with `question`, gg`context`, `answer`) — visible only by manually reading CloudWatch, and the retrieved `context` is never persisted to the DB, so it's lost once the log line scrolls out of easy reach. The user wants a simple `/monitoring` page now (linked from the existing `/admin/maintenance` console) as an interim way to inspect recent chat activity and eval data, before wiring up Grafana + Prometheus for proper dashboards/metrics later.

Decisions from the user:

- Data source: **both** — keep using the DB (`ChatLog` table) as the primary source for the page's table/stats, **and** persist the retrieved `context` going forward (currently only printed, never stored) so eval data isn't lost. Live CloudWatch querying via boto3 is also wanted as a secondary "raw log tail" capability.
- Page content: a **recent chat log table** (question, answer, category, topic, response_time, unanswered flag, and now retrieved context).
- Auth: reuse **`get_current_admin`** (HTTPBasic), same as the rest of `/admin/*`, consistent with where it's linked from (`maintenance.html`).

## Plan

### 1. Persist retrieved context (closes the eval data gap)

- `models.py`: add `retrieved_context = Column(Text, nullable=True)` to `ChatLog` (`models.py:6-17`).
- `main.py` `run_migrations()`: add `("chat_logs", "retrieved_context", "TEXT")` to the migration `columns` list, per the existing `ALTER TABLE ... ADD COLUMN` pattern (CLAUDE.md: never drop/rename, only add).
- `core_ai.py:558-567` (where `new_log = models.ChatLog(...)` is built, right before the `RAG_EVAL` print at line 570): set `retrieved_context=docs` (the same `docs` value already used in `eval_data["context"]`) so it's saved once instead of only printed. `docs` needs to be stringified if it isn't already a string (check its type where `eval_data` is built — join/stringify consistent with what's printed).

### 2. New `/admin/monitoring` route + template

- `main.py`, near `kpi_dashboard` (`main.py:249-301`) which is the closest existing analog: add `GET /admin/monitoring` guarded by `Depends(get_current_admin)`.
  - Query last N (e.g. 50-100) `ChatLog` rows ordered by `timestamp.desc()` — mirror the query pattern already used in `kpi_dashboard`.
  - Include per-row: timestamp, session_id, user_query, bot_answer, category, topic, response_time, is_unanswered, retrieved_context (from step 1).
  - Include the same aggregate stats already computed in `kpi_dashboard` (total_chats, avg_speed_all, avg_speed_last_10) plus an unanswered-rate count, so the page also works as a quick health snapshot.
- New template `templates/monitoring.html`, following the visual conventions of `maintenance.html`/`kpi.html` (Bootstrap 5.3 CDN, Font Awesome, Cairo font, RTL Arabic, inline `<style>`/`<script>`, no custom JS file needed) — table of recent logs (highlight `is_unanswered=True` rows), stat cards at top, and an expandable/collapsible cell or modal for `retrieved_context` and `bot_answer` (can be long) reusing the `chat_details.html` pattern for a single-record detail view if useful.

### 3. Live CloudWatch tail (secondary capability)

- Add `boto3` to `image/pyproject.toml` and `image/requirements.txt` dependencies.
- `main.py` or a small helper in `core_ai.py`: a function that uses `boto3.client("logs")` to fetch the most recent N events from log group `/ecs/sstli-chatbot` (via `filter_log_events` or `get_log_events` on the latest stream), parse lines starting with `{"log_type": "RAG_EVAL"...}` as JSON, and return them.
  - This will only work when running with valid AWS credentials/IAM permissions (i.e., on the Fargate task role, or locally with configured AWS creds) — wrap in try/except so local dev without AWS access doesn't break the page, just shows "CloudWatch unavailable" instead of erroring.
- On `/admin/monitoring`, add a secondary section/tab "Live CloudWatch (last N)" that calls this helper — separate from the DB-backed table so one failing doesn't break the other.

### 4. Link from maintenance page

- `templates/maintenance.html:142-154` (navbar): add a button next to the existing "الإحصائيات" (`/admin/kpi`) and "Track Dashboard" links: `<a href="/admin/monitoring" class="btn ...">مراقبة النظام (Monitoring)</a>`.

## Files to modify

- `image/src/rag_app/models.py` — add `retrieved_context` column
- `image/src/rag_app/main.py` — migration entry, new `/admin/monitoring` route, CloudWatch helper (or import from core_ai.py)
- `image/src/rag_app/core_ai.py` — set `retrieved_context` on `ChatLog` creation
- `image/src/rag_app/templates/monitoring.html` — new template
- `image/src/rag_app/templates/maintenance.html` — add nav link
- `image/pyproject.toml`, `image/requirements.txt` — add `boto3`

## Verification

1. Run locally (`uvicorn main:app --reload`), send a few chat messages, confirm `run_migrations()` adds the new column without error and `ChatLog.retrieved_context` gets populated.
2. Visit `/admin/monitoring` with admin Basic Auth credentials; confirm the table renders recent logs including context, and stat cards match `/admin/kpi` numbers.
3. Confirm the CloudWatch section gracefully shows "unavailable" locally (no AWS creds) without crashing the page.
4. After deploying, confirm the CloudWatch section pulls real `RAG_EVAL` events from `/ecs/sstli-chatbot` (requires the Fargate task role to have `logs:FilterLogEvents`/`logs:GetLogEvents` permission — flag to the user if IAM needs updating).
5. Click through from `/admin/maintenance` to confirm the new nav link works.
