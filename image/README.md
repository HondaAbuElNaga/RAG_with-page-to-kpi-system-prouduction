## 1. Run Locally (Python)

If you want to test without Docker:

`cd image`
#### Unix/macOS:

`source .venv/bin/activate`
#### Windows:

`.venv\Scripts\activate`

#### Install dependencies

`pip install -r requirements.txt`

#### Run the server

`cd image/src/rag_app`
`uvicorn main:app --reload --host 0.0.0.0 --port 8000`

#### Using uv

`uv sync`
`cd src/rag_app`
`uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000`


## 2. Docker image

### Build Docker image

`docker builder prune -f`
`docker build -t my-rag-app .`

### Run docker image for testing

`docker-compose up`

```bash
    docker run -p 8080:80 --env-file .env
    -e CHROMA_PATH=/data/chroma_db `
    -e DB_PATH=/data/kpi_data.db `
    -v "E:\Machine Learning\CodE\2.
    Projects\prouduction\image\src\rag_app\data:/data" `
    my-rag-ap
```

## 3. Deployment to AWS (ECS & ECR)

### Push to dockerhub

`docker build -t ebrahemhesham/rag-app:v2 ./image`
`docker login`
`docker push ebrahemhesham/rag-app:v2`

### Update AWS ECS Service

```bash
aws ecs update-service \
    --cluster default \
    --service sstli-chatbot-spot \
    --task-definition sstli-chatbot-v3 \
    --force-new-deployment
```

#### Create EFS (Repeat for each subnet ID where your Fargate tasks run)

```bash
aws efs create-mount-target \
    --file-system-id <fs-id> \
    --subnet-id <subnet-id> \
    --security-groups <security-group-id>
```

 ## 4. to AWS CloudWatch Logs Evluated

fields @timestamp, question, context, answer
| filter log_type = "RAG_EVAL"
| sort @timestamp desc

W62ZTthTZ6A6prn
https://d14hbi7dyty7wy.cloudfront.net/chat
python -c "import sqlite3; conn = sqlite3.connect('kpi_data.db'); conn.execute('ALTER TABLE uploaded_reports ADD COLUMN report_period VARCHAR'); conn.execute('ALTER TABLE uploaded_reports ADD COLUMN period_label VARCHAR'); conn.commit(); conn.close(); print('Done!')"
sqlite3 kpi_data.db "DELETE FROM uploaded_reports WHERE id IN (4);
sqlite3 kpi_data.db "DELETE FROM weekly_notes;"

Sales: https://d14hbi7dyty7wy.cloudfront.net/dashboard
Admin (Track): https://d14hbi7dyty7wy.cloudfront.net/trackdashboard
Login: https://d14hbi7dyty7wy.cloudfront.net/dashboard/login

