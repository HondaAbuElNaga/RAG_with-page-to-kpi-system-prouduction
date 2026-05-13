# RAG Chatbot System — Project Guide

## Stack
- **Backend**: FastAPI + Uvicorn
- **AI**: LangChain + OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Vector DB**: ChromaDB (persisted at `CHROMA_PATH`, env-overridable)
- **App DB**: SQLite via SQLAlchemy ORM (`kpi_data.db`, `sql_app.db`)
- **Auth**: Cookie-based sessions (`dashboard_session`), SHA-256 passwords
- **Deploy**: Docker → DockerHub → AWS ECS/ECR + EFS for ChromaDB persistence

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
  data/chroma_db/— ChromaDB vector store (uploaded via admin panel)
```

## Run Locally
```bash
# activate venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Unix

# run server
cd image/src/rag_app
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# or with uv
cd image && uv sync
cd src/rag_app && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Docker
```bash
docker build -t my-rag-app .
docker run -p 80:80 --env-file .env my-rag-app
```

## Deploy to AWS
```bash
docker tag my-rag-app ebrahemhesham/rag-app:v1
docker push ebrahemhesham/rag-app:v1
aws ecs update-service --cluster default --service sstli-chatbot-spot --force-new-deployment
```

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
- Wrap each migration in try/except — idempotent, safe to re-run.
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
| `CHROMA_PATH` | Optional — override ChromaDB path (used for EFS mount in ECS) |
| `ADMIN_USER` / `ADMIN_PASS` | Basic auth for `/admin/*` routes |

## External URLs (Production)
- Chat (AR): `/`  
- Chat (EN): `/chat-en`
- Sales Dashboard: `/dashboard`
- Admin Track: `/trackdashboard`
- Login: `/dashboard/login`
