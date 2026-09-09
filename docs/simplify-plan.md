# Simplify pass — whole codebase (safe fixes only)

## Context
A `/simplify`-style review was run across the whole codebase (not just a diff) using three parallel Explore agents covering (1) `main.py`/`core_ai.py`, (2) `models.py`/`schemas.py`/`auth.py`/`database.py`, and (3) `templates/`/`static/js/`. The scan surfaced both small low-risk cleanups and larger structural refactors (merging AR/EN templates, extracting a shared `base.html`, adding ORM mixins, a shared JS fetch helper). The user chose **safe fixes only** — no visual changes, no cross-file structural refactors, nothing that could alter behavior. This plan lists only the fixes that meet that bar. Larger refactors found during the scan are listed at the bottom as explicitly out of scope for this pass.

## Fixes to apply

### `image/src/rag_app/core_ai.py`
1. **Delete dead code**: the fully commented-out `generate_session_summary_background` function (~lines 359-408) and its commented call site (~line 581).
2. **Remove duplicate imports**: `os` is imported twice (line 1 and ~19); `engine, SessionLocal` imported twice (~18 and ~20). Keep one of each.
3. **Fix blocking call in async path**: `prepare_rag_context` (~line 454) calls `llm.invoke(...)` for the rephrase step inside an otherwise-async pipeline (`generate_response_stream`). Change to `await llm.ainvoke(...)` to match the rest of the pipeline — no behavior change, removes a blocking call from the async flow.

### `image/src/rag_app/main.py`
4. **Consolidate duplicated month/day-label constants**: `_MONTHS_AR` already exists (~line 61) but Arabic month lists are redefined inline at ~1267-1268, ~1279-1280, ~1642-1643; `days_ar` is redefined at ~1576 and ~1641. Replace all inline redefinitions with references to the existing module-level constants (add a `_DAYS_AR` constant alongside `_MONTHS_AR` if one doesn't already exist).
5. **Deduplicate period-label logic**: in `upload_report_for_request`, the `peak_hours` and `repeated_visitors` branches (~lines 1263-1289) are near-identical copy-paste differing only in the "daily" option. Extract a small `_build_period_label(report_type, filename, req)` helper and call it from both branches.
6. **Fix redundant SQL-then-Python-filter pattern**: the "pull all rows, then filter by `timestamp.isocalendar()[1] == week` in Python" pattern is repeated at ~88-99, ~1573-1583, ~1638-1650, ~1704-1717, pulling entire tables into memory each time. `get_week_data` (~1036-1043) already does this correctly with a SQL `BETWEEN`/range filter — apply the same SQL-range approach at the other call sites instead of the Python `isocalendar()` filter.
7. **Move blocking disk I/O off the event loop**: `upload_db` (~181-193, uses `open()`, `shutil.copyfileobj`, `shutil.rmtree`, `zipfile.ZipFile.extractall`, `os.remove`) and `upload_pdfs`'s call into `ingest_pdfs` (~line 234, CPU/disk-bound PDF parsing + embedding) run synchronous work directly inside `async def` routes, blocking the event loop for other requests. Wrap the blocking calls with `starlette.concurrency.run_in_threadpool` (already a FastAPI dependency, no new package needed).
8. **Log swallowed migration errors**: `run_migrations()` (~912-941) catches `Exception` per-column and silently ignores it (intended for "column already exists"). Add a log line with the exception message so real failures (e.g. connection errors) aren't silently invisible, without changing the idempotent try/except structure.

### `image/src/rag_app/auth.py`
9. **Deduplicate dashboard/trackdashboard auth**: `get_trackdashboard_user` (~59-67) re-implements the same cookie-extraction + missing-token-redirect logic as `get_dashboard_user` (~52-56), then adds a role check. Refactor `get_trackdashboard_user` to depend on `get_dashboard_user` (`Depends(get_dashboard_user)`) and only add the role check on top, removing the duplicated cookie/redirect logic.

### `image/src/rag_app/database.py`
10. **Fix deprecated import**: `from sqlalchemy.ext.declarative import declarative_base` → `from sqlalchemy.orm import declarative_base`. Same symbol, no behavior change, removes a deprecation warning.

### `image/src/rag_app/static/js/chat.js`
11. **Delete dead no-op function**: `updateLeadKeywords()` (~lines 32-34) does nothing but is still called on every message (~line 282). Delete the function and its call site.
12. **Compute `isEnglish` once**: `showPhoneForm()` and `addBotGreeting()` each independently recompute `window.location.pathname.includes('/chat-en')` (~lines 97, 253, 334). Compute it once in the `DOMContentLoaded` handler and pass/reuse it instead of recomputing.

## Explicitly out of scope for this pass
(Found during the scan, but skipped as structural/behavior-risk — not part of "safe fixes only")
- Merging `chat.html`/`chat_en.html` and `user_maintenance.html`/`_en.html` into single locale-parameterized templates.
- Extracting a shared `base.html`/sidebar include across `dashboard.html`, `leads_report.html`, `kpi.html`, `trackdashboard.html`, `maintenance.html`.
- SQLAlchemy `IDMixin`/`TimestampMixin` for `models.py`, and switching `datetime.utcnow()` → timezone-aware `datetime.now(timezone.utc)` (risks naive/aware datetime comparison breaks elsewhere in the codebase).
- Shared JS `apiFetch()` helper to replace ~25 inline `fetch()` blocks across dashboard/trackdashboard/leads_report/kpi templates.
- Standardizing inline `get_dashboard_user(request)` calls to `Depends(...)` across all routes (touches many route signatures — mechanical but broad).
- Extracting a shared system-preamble constant across the LLM prompts in `core_ai.py` (risk of subtly changing model behavior).
- Extracting a shared `_parse_llm_json`/code-fence-stripping helper (only one call site today — premature per "don't abstract before three uses").

## Verification
- Run the app locally (`uvicorn main:app --reload` from `image/src/rag_app`) and exercise:
  - Chat flow (`/` and `/chat-en`) to confirm the async rephrase fix and chat.js dead-code removal don't break the phone-form / greeting flow.
  - Admin upload DB (`/admin/upload-db`) and PDF ingestion to confirm `run_in_threadpool` wrapping still works end-to-end.
  - Dashboard/trackdashboard login and a weekly report view to confirm the SQL-range week-filter fix returns the same data as before.
  - Sales dashboard report generation (peak hours / repeated visitors) to confirm the extracted `_build_period_label` helper produces identical labels.
- Run any existing test suite if present (check for `pytest`/`tests/` — none was confirmed during exploration, verify at implementation time).
- `git diff` review before committing to confirm no unintended template/JS/CSS changes crept in.
