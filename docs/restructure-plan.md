# Restructure Plan — Backend Split + Separate Frontend

Status: proposed, not started
Date: 2026-09-09
Scope: `image/src/rag_app/` → layered backend package + standalone frontend projects

---

## Why

The app works, but three things are fused together that shouldn't be:

1. `main.py` is one 1946-line module holding routes, migrations, CloudWatch queries,
   CSV export and report building.
2. The "frontend" is 14 Jinja templates with all CSS and JS inlined — there is no
   frontend project, no build step, and no way to reuse anything between pages.
3. Local paths, Docker paths and ECS/EFS paths are configured in three different
   places with three different conventions.

---

## What the exploration found

| Finding | Detail |
|---|---|
| `main.py` | 1946 lines — routes + migrations + CloudWatch + CSV + reports |
| `core_ai.py` | 540 lines — RAG engine, lead scoring and background tasks mixed |
| Templates | 14 files, **5,776 lines** of HTML |
| Inline assets | **Every** template has inline `<style>` *and* inline `<script>` |
| External static | `url_for('static')` used **zero** times; only `chat.html` / `chat_en.html` reference `/static/js/chat.js`, hardcoded |
| Biggest templates | `trackdashboard.html` 1452, `dashboard.html` 1215, `leads_report.html` 824 |
| API surface | **26 distinct `fetch()` endpoints** already called from template JS |
| Audiences | Public chat (`/`, `/chat-en`, Arabic-first RTL) vs. internal dashboards (7+ pages, cookie auth) |
| Hosting | ECS Fargate, single container on port 80 behind an ALB, EFS mounted at `/data` |

The key insight: **the dashboards are already SPAs in disguise.** The JSON API mostly
exists — it's just entangled with server-rendered HTML. This is a decoupling job, not
a rewrite.

---

## Target structure

```
prouduction/
├── backend/                       ← was image/
│   ├── Dockerfile  pyproject.toml  uv.lock
│   └── app/
│       ├── main.py                app factory + middleware only (~80 lines)
│       ├── config.py              ONE Settings object: all env vars + path resolution
│       ├── db/                    session.py  models.py  migrations.py
│       ├── api/
│       │   ├── public.py          chat, lead submit, health
│       │   ├── admin.py           kpi, upload-db, upload-pdfs, maintenance
│       │   ├── dashboard.py       sales dashboard + week data
│       │   ├── track.py           trackdashboard, users, sections, notes
│       │   └── reports.py         exports, peak/leads reports
│       ├── services/              rag.py  scoring.py  cloudwatch.py  ← splits core_ai.py
│       ├── schemas/  auth/
│       └── web/                   thin Jinja shells only (no inline CSS/JS)
├── frontend/
│   ├── chat/                      public widget (AR + EN) — vanilla or Preact, tiny
│   ├── admin/                     dashboards SPA (Vite + chosen framework)
│   └── shared/                    design tokens, API client, RTL/i18n
├── infra/                         compose, ECS task defs, deploy scripts ← from docs/
└── data/                          gitignored; local mirror of EFS /data
```

---

## Phasing

Four independently shippable steps. No big-bang cutover — each phase leaves the app
deployable.

### Phase 0 — config consolidation (½ day)

One `config.py` owning `DB_PATH`, `CHROMA_PATH`, `OPENAI_API_KEY`, admin credentials
and CORS origins.

The CWD-independent path resolution already applied to `database.py` and `core_ai.py`
(2026-09-09) collapses into two lines there.

No behaviour change. **Do this first — everything else depends on it.**

### Phase 1 — backend split (2–3 days)

`main.py` → `APIRouter` modules along the boundaries above. Purely mechanical: zero URL
changes, so the frontend cannot notice.

Add real `__init__.py` files and drop the `PYTHONPATH=/app/src:/app/src/rag_app` hack in
the Dockerfile — that dual-path is what forces flat `import models` and blocks any proper
package layout.

Also convert the two `@app.on_event("startup")` handlers (`run_migrations`,
`seed_default_sections`) to a `lifespan` handler; `on_event` is deprecated in this
FastAPI version.

### Phase 2 — frontend extraction (~1 week)

The actual goal. For each template:

1. Lift inline `<style>` / `<script>` into `frontend/<app>/` sources.
2. Build with Vite, output to `backend/app/static/dist/`.
3. Jinja keeps rendering a **shell** — `<div id="root">` plus a JSON
   `window.__BOOTSTRAP__` blob — so auth, routing and the ALB are untouched.

Order: start with `monitoring.html` (248 lines, newest, lowest traffic) as the pilot,
then `kpi.html`, then the two 1200+ line dashboards last.

### Phase 3 — optional origin split

Only if CDN caching is wanted. See Option B below.

---

## AWS hosting: pick one

### Option A — single container, frontend built at image build time ← **recommended**

Multi-stage Dockerfile: `node:20 → npm run build → COPY dist` into the Python stage,
served by `StaticFiles`.

- Same ECS service, same ALB, same EFS.
- No CORS changes, no cookie changes, no new AWS resources, **no added cost**.
- Image grows ~50 MB; build time +1–2 min.

### Option B — S3 + CloudFront for frontend, ALB→ECS for `/api`

Better caching, cheaper egress, independent deploys. But it first costs you two fixes
that are currently landmines:

- CORS is `allow_origins=["*"]`. Combined with `allow_credentials=True` browsers reject
  it outright, and with cookie auth it is a genuine risk. Exact origins must be pinned
  first.
- `dashboard_session` is `httponly` + `samesite=lax`. Cross-origin it needs
  `SameSite=None; Secure`, plus `credentials: 'include'` on all 26 fetch calls.

**Decision: go with A.** Phase 2 leaves you one `npm run build` away from B whenever you
want it, and B's benefits are marginal for an internal tool with a handful of dashboard
users.

---

## Risks to settle before starting

1. **`asyncio.create_task` background jobs + multiple ECS tasks.** Lead scoring runs
   in-process. Scaling `desired-count` past 1 gives SQLite-on-EFS concurrent writers,
   which will corrupt it. The restructure is the right moment to decide: stay at one
   task, or move to RDS/Postgres.
2. **Deprecated startup events** — handled in Phase 1, noted here so it isn't lost.
3. **Arabic RTL is baked into the inline CSS of every template.** Extract shared
   RTL/direction handling into `frontend/shared` during Phase 2 or it gets duplicated
   14 times.
4. **Moving `image/` → `backend/` changes the Docker build context.**
   `docker-compose.yaml`, both ECS task definitions under `docs/`, and the
   `docker build` commands in `CLAUDE.md` must all change in the same commit.

---

## Related open item

Two divergent local databases (not production — production is on EFS):

| | `rag_app/kpi_data.db` | `rag_app/data/kpi_data.db` |
|---|---|---|
| chat_logs | **229** | 3 |
| leads | **106** | 2 |
| dashboard_users | 2 | 1 |
| latest log | **2026-09-09** | 2026-08-04 |

`rag_app/kpi_data.db` is the canonical local dev DB. The `data/` copy is a leftover from
a `docker-compose up` smoke test. Recommendation: archive the `data/` copy and point
compose's `DB_PATH` at `/data/kpi_data.dev.db` so the volume mount can never shadow the
real dev database. **Not yet actioned.**
