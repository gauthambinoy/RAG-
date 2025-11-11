# Cloud Deployment Guide

This project ships with a Dockerfile and a FastAPI server. Below are three quick deployment options.

## Option A: Render (free tier)

1. Push your repo to GitHub.
2. Create a new Web Service in Render:
   - Connect GitHub repo.
   - Runtime: Docker
   - Build Command: (leave empty, Docker builds automatically)
   - Start Command: `uvicorn app.api:app --host 0.0.0.0 --port 8000`
   - Add Environment Variable: `GEMINI_API_KEY=...`
3. Click Deploy. When live, visit the URL and check `/docs` and `/health`.

Notes:
- Persisted indices: mount a persistent disk or rebuild on startup by running `scripts/build_index.py` in a pre-start hook.

## Option B: Azure Container Apps

1. Build and push image

```bash
# Build
docker build -t your-docker-username/rag-system:latest .
# Login
az acr login --name <your-acr-name>
# Tag and push
docker tag your-docker-username/rag-system:latest <your-acr-name>.azurecr.io/rag-system:latest
docker push <your-acr-name>.azurecr.io/rag-system:latest
```

2. Deploy container app

```bash
az containerapp create \
  --name rag-system \
  --resource-group <rg> \
  --environment <env> \
  --image <your-acr-name>.azurecr.io/rag-system:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars GEMINI_API_KEY=your_gemini_key_here
```

3. Test

```bash
curl -s https://<app-domain>/health
```

## Option C: AWS App Runner (recommended)

No servers to manage. Uses ECR for images and App Runner for HTTPS, autoscaling, health checks.

Prereqs
- AWS account and region (e.g., us-east-1)
- ECR repository (e.g., rag-service)
- Use AWS SSO for CLI (no long-term access keys):
  - `aws configure sso`; `aws sso login`

Build and push image to ECR
```bash
# variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<your-12-digit-account-id>
export ECR_REPO=rag-service
export IMAGE_TAG=main-$(date +%Y%m%d%H%M)

# create repo (idempotent)
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" || true

# login to ECR
aws ecr get-login-password --region "$AWS_REGION" \
| docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com

# build, tag, push
docker build -t "$ECR_REPO:$IMAGE_TAG" .
docker tag "$ECR_REPO:$IMAGE_TAG" "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO:$IMAGE_TAG"
docker push "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO:$IMAGE_TAG"
```

Create App Runner service (Console)
- App Runner → Create service → Source: Amazon ECR → pick your image tag
- Port: 8000; Health check path: /health
- Environment variables: set GEMINI_API_KEY (required), optional MODEL_NAME=gemini-2.5-flash
- Auto deploy from ECR: On
- After Running, test: https://<apprunner-url>/health and https://<apprunner-url>/docs

Optional: Custom domain + HTTPS
- Route 53 hosted zone for your domain
- ACM certificate in same region → DNS validate
- App Runner → Custom domains → attach cert → add CNAME → wait Ready

## Health Verification

Locally, you can verify the API and pipeline without starting a server using:

```bash
python scripts/health_check.py
```

It initializes the FastAPI app, loads the saved indices under `outputs/`, and checks `/health` and `/stats`. Exit code 0 indicates readiness.

## Tips

- Ensure `outputs/embeddings` is available or run `python scripts/build_index.py` at least once with your documents in `data/raw_documents`.
- Set `RAG_ENABLE_RERANKER=1` if you want higher precision results (adds latency).
- Use `STREAMLIT_SERVER_PORT=8501` to run the Streamlit dashboard in cloud environments where default ports are blocked.
