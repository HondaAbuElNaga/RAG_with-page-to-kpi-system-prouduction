# RAG Chatbot System — Project Guide

## Stack
- **Backend**: FastAPI + Uvicorn
- **AI**: LangChain + OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Vector DB**: ChromaDB (persisted at `CHROMA_PATH`, env-overridable)
- **App DB**: SQLite via SQLAlchemy ORM (`data/kpi_data.db`)
- **Auth**: Cookie-based sessions (`dashboard_session`), SHA-256 passwords
- **Deploy**: Docker → DockerHub → AWS ECS/Fargate; EFS mounted at `/mnt/efs` persists both the SQLite DB and ChromaDB

## Project Layout
```
image/src/rag_app/
  main.py        — FastAPI routes (admin, sales dashboard, leads, reports)
  core_ai.py     — RAG engine, vector store, background LLM tasks
  models.py      — SQLAlchemy ORM models
  schemas.py     — Pydantic request/response schemas
  database.py    — DB engine + session factory
  auth.py        — Admin/dashboard auth helpers
  templates/     — Jinja2 HTML templates
  static/js/     — Frontend JS (chat.js)
  data/          — persistent state; ECS mounts EFS over this at /mnt/efs
    chroma_db/   — ChromaDB vector store (uploaded via admin panel)
    kpi_data.db  — canonical SQLite app DB
```

See `docs/restructure-plan.md` for the proposed backend/frontend split.

## Run Locally
```bash
cd image && uv sync
cd src/rag_app && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
`image/.venv` is a **Windows** venv (uv-managed, CPython 3.12). It does not work under
WSL — for that, build a separate one with `UV_PROJECT_ENVIRONMENT=.venv-linux uv sync`.

`requirements.txt` is a uv export; `uv sync` is the supported install path.

## Docker
All Docker commands run **from the repo root**, not `image/`. `docker-compose.yaml` owns
the build context, image tag (`$TAG`, default `v3`), port, env, volume and healthcheck,
so the image tested locally is the one pushed.
```bash
docker compose build      # tags ebrahemhesham/rag-app:v3
docker compose up         # http://localhost:8081
```

## Deploy to AWS
Set `$TAG` once per shell — inline `TAG=v4 docker ...` is bash-only and fails in cmd
(`set TAG=v4`) and PowerShell (`$env:TAG = "v4"`). Order matters: `build` creates the
tag, `push` uploads it.
```
set TAG=v4
docker compose build      # 1. build + tag
docker compose up -d      # 2. verify on http://localhost:8081
docker login              # 3.
docker compose push       # 4. upload
# then bump the image tag in the task definition and redeploy:
aws ecs update-service --cluster default --service sstli-chatbot-spot --force-new-deployment
```
Compose builds for the host platform; Fargate runs `LINUX/X86_64`. Building from an ARM
Mac needs `platform: linux/amd64` on the service.

---

## RAG Best Practices (This System)

### Vector Store
- `SIMILARITY_THRESHOLD = 1.5` — only docs with score < 1.5 pass. Tune per domain corpus.
- `TOP_K_RESULTS = 5` — retrieve 5, filter by threshold, pass survivors to LLM.
- `MEMORY_WINDOW_SIZE = 3` — last 3 turns for context. Don't increase; cost grows fast.
- ChromaDB collection name: `"example_collection"` — keep consistent across upload/query.
- Upload new KB: zip the `chroma_db/` folder, upload via `/admin/upload-db`.

### Query Rephrasing
- When history exists, rephrase the user query before similarity search (see `prepare_rag_context`).
- Keep rephrased query in the **same language as the user message** — this is enforced in the rephrase prompt.
- If rephrase fails, fall back to raw message — never block the response.

### Prompt Engineering
- Inject `GLOBAL_FACTS` (branch locations) at the top of every RAG prompt — static facts should not rely on retrieval.
- Detect user language in every turn; enforce language match in system instructions.
- Keep system instructions short and numbered — LLMs follow numbered lists better.
- For structured extraction (intent, categories): demand JSON output, strip code fences before parsing.

### Lead Scoring
- Scoring runs in background via `asyncio.create_task` — never blocks streaming response.
- `hot` = asked price OR asked registration. `warm` = 5+ questions. `cold` = otherwise.
- Intent detection re-analyzes ALL user messages in session, not just the latest — avoids missing delayed signals.
- Early-exit if both intents already detected to save OpenAI tokens.

### Streaming
- Use `llm.astream()` for chat responses — yields tokens as they arrive.
- Log `first_token_time` (time to first token), not total response time — better latency metric.
- Detect unanswered responses by scanning bot output for apology phrases (AR + EN).

### Database Migrations
- Schema changes are applied at startup via `ALTER TABLE ... ADD COLUMN` in `run_migrations()`.
- `run_migrations()` reads `PRAGMA table_info` per table first and only ALTERs columns that
  are genuinely missing, so a current schema prints one line, not 17 caught exceptions.
- A failed ALTER now prints `✗ ... FAILED` — treat it as a real error, not noise.
- Never drop or rename columns in migrations — only ADD new ones.

### Auth
- Admin routes: Basic Auth (`get_current_admin`).
- Dashboard/trackdashboard: Cookie session (`get_dashboard_user` / `get_trackdashboard_user`).
- Session cookie: `httponly=True`, `samesite=lax`, 8-hour expiry.
- Passwords: SHA-256 hash, no salt — acceptable for internal tool, not for public auth.

### CORS
- Currently `allow_origins=["*"]` — restrict to actual frontend domain before exposing sensitive routes.

### CloudWatch Evaluation
- Every response emits a JSON log line with `log_type: "RAG_EVAL"`, question, context, answer.
- Query in CloudWatch Logs Insights:
  ```
  fields @timestamp, question, context, answer
  | filter log_type = "RAG_EVAL"
  | sort @timestamp desc
  ```

---

## Key Patterns to Follow

**Adding a new route**: add to `main.py`, use `Depends(get_db)` for DB, call `get_dashboard_user` or `get_trackdashboard_user` for auth.

**Adding a new LLM background task**: create `async def my_task(...)`, call with `asyncio.create_task(my_task(...))` after the streaming response finishes. Always `db.close()` in `finally`.

**Updating the knowledge base**: zip `chroma_db/` directory, POST to `/admin/upload-db`. The old DB is deleted, new one extracted, vector store reloaded.

**Adding a DB column**: add to the `columns` list in `run_migrations()` in `main.py`. Format: `(table_name, column_name, sql_type)`.

**Changing LLM model**: update `llm` and `embeddings_model` in `core_ai.py`. If changing embeddings model, rebuild ChromaDB — old embeddings are incompatible.

---

## Environment Variables
| Var | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | Required — OpenAI API access |
| `DB_PATH` | SQLite file. ECS: `/mnt/efs/kpi_data.db`. Local default: `data/kpi_data.db` |
| `CHROMA_PATH` | ChromaDB dir. ECS: `/mnt/efs/chroma_db`. Local default: `data/chroma_db` |
| `ADMIN_USER` / `ADMIN_PASS` | Basic auth for `/admin/*` routes |

**Path resolution**: relative `DB_PATH`/`CHROMA_PATH` values resolve against the
application directory (`image/src/rag_app/`), never the shell's working directory — the
CWD you launch uvicorn from cannot change which database you open. Absolute paths
(as ECS uses) are taken as-is.

**Canonical local DB**: `image/src/rag_app/data/kpi_data.db`. That folder is what the
EFS volume mounts over in ECS and what compose bind-mounts to `/mnt/efs`, so local runs
and containers share one location.

⚠️ Secrets are currently plaintext in `image/.env` and in the `environment` blocks of
both task definitions under `docs/`. Move them to Secrets Manager (`secrets` +
`valueFrom`) before wider exposure.

## External URLs (Production)
- Chat (AR): `/`  
- Chat (EN): `/chat-en`
- Sales Dashboard: `/dashboard`
- Admin Track: `/trackdashboard`
- Login: `/dashboard/login`
