# Cloud Deployment Guide

This project ships with a Dockerfile and a FastAPI server. Below are two quick deployment options.

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
