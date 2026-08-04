## 1. Run Locally (Python)

If you want to test without Docker:

```bash
cd image
```

#### Unix/macOS
```bash
source .venv/bin/activate
```

#### Windows
```powershell
.venv\Scripts\activate
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Run the server
```bash
cd src/rag_app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Using uv (recommended)
```bash
uv sync
cd src/rag_app
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 2. Docker image

### Build

```bash
docker build -t my-rag-app .
```

### Run locally for testing

**Preferred: docker-compose** (from the repo root, not `image/`) — `docker-compose.yaml` already wires up the port, env vars, data volume mount, and a healthcheck:

```bash
docker-compose up
```

**Manual alternative**, if you don't want compose (from `image/`):

```bash
docker run -p 8080:80 --env-file .env \
    -e CHROMA_PATH=/data/chroma_db \
    -e DB_PATH=/data/kpi_data.db \
    -v "$(pwd)/src/rag_app/data:/data" \
    my-rag-app
```

On Windows PowerShell, replace `$(pwd)/src/rag_app/data` with the absolute path to `image/src/rag_app/data`, and the trailing `\` line continuations with `` ` ``.

## 3. Deployment to AWS (ECS & ECR)

### Push to DockerHub

Bump the version tag each release:

```bash
docker build -t ebrahemhesham/rag-app:v3 ./image
docker login
docker push ebrahemhesham/rag-app:v3
```

### Run the versioned image (sanity check before/after pushing)

From the repo root — same command whether you just built it locally or pulled it back down from DockerHub (`docker pull ebrahemhesham/rag-app:v3` first, in the latter case):

```bash
docker run -p 8080:80 --env-file ./image/.env \
    -e CHROMA_PATH=/data/chroma_db \
    -e DB_PATH=/data/kpi_data.db \
    -v "$(pwd)/image/src/rag_app/data:/data" \
    ebrahemhesham/rag-app:v3
```

Visit `http://localhost:8080` and confirm the app comes up before updating the ECS service.

### Update AWS ECS service

Update the task definition to reference the new image tag first, then:

```bash
aws ecs update-service --cluster default --service sstli-chatbot-spot --task-definition sstli-chatbot-v3 --force-new-deployment
```

### Create EFS mount targets (repeat for each subnet Fargate tasks run in)

```bash
aws efs create-mount-target \
    --file-system-id <fs-id> \
    --subnet-id <subnet-id> \
    --security-groups <security-group-id>
```

## 4. CloudWatch Logs Insights query (RAG eval)

```
fields @timestamp, question, context, answer
| filter log_type = "RAG_EVAL"
| sort @timestamp desc
```

W62ZTthTZ6A6prn

## Production URLs

- Chat: https://d14hbi7dyty7wy.cloudfront.net/chat
- Sales dashboard: https://d14hbi7dyty7wy.cloudfront.net/dashboard
- Admin (Track): https://d14hbi7dyty7wy.cloudfront.net/trackdashboard
- Login: https://d14hbi7dyty7wy.cloudfront.net/dashboard/login
