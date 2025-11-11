# Submission Overview

This document points graders to the exact artifacts for evaluation and hides any non-essential folders.

## What to Run

- Build index (one-time): `python scripts/build_index.py --dense`
- Streamlit UI: `streamlit run app/dashboard.py`
- FastAPI API: `uvicorn app.api:app --host 0.0.0.0 --port 8000`
- Health (no server needed): `python scripts/health_check.py`

Set `GEMINI_API_KEY` in a `.env` file (see `.env.example`).

## Where to Look

- Data prep & chunking: `src/preprocessing/`
- Retrieval (FAISS, BM25, hybrid, reranker flag): `src/retrieval/`
- LLM interface (Gemini-only): `src/generation/llm_interface.py`
- Pipeline glue: `src/pipeline.py`
- Test queries: `src/queries/test_queries.py`
- Evaluation scripts: `scripts/run_evaluation.py`, `scripts/evaluation_report_v2.py`
- API/UI: `app/api.py`, `app/dashboard.py`
- Audit report: `REPORT.md`

## Deliverables (Challenge Mapping)

- Data Preparation: loaders + normalization + chunking decisions documented in `REPORT.md`.
- Test Queries: 20 diverse queries with rationale in `src/queries/test_queries.py`.
- Retrieval: Dense + optional BM25 hybrid and RRF; reranker toggled via `RAG_ENABLE_RERANKER`.
- Generation: Gemini-only with context grounding; default model `gemini-2.5-flash`.
- Evaluation: Multiple criteria + hallucination report (`outputs/metrics_report_v2.json`).
- Deployment: Dockerfile, `DEPLOY.md` quick guide.

## Ignore These (not part of grading)

- `RAG-/` (archival duplicate of the project)
- `__pycache__/`, `.venv/`
- `outputs/embeddings/models/` (downloaded model cache)
- `outputs/logs/`

## Live Demo URL

Provide your deployed link here:

- API: https://<your-app-domain>/docs
- UI:  https://<your-ui-domain>/

### Deployment Status (AWS App Runner)

- ECR Repo: `rag` (region `eu-west-1`)
- Latest Image Tag: generated per commit (`main-<sha>`) via `ecr-push` workflow
- CloudFormation Stack: `rag-apprunner` (deploy with `deploy-apprunner` workflow)
- Next Manual Step: Run "Deploy App Runner" workflow once OIDC role & `GEMINI_API_KEY` secret are in place
- After successful deploy: replace placeholders above with actual service URL

Checklist Remaining:
1. Set repo variable `AWS_ROLE_TO_ASSUME` with IAM role ARN
2. Add secret `GEMINI_API_KEY`
3. Push/dispatch workflows
4. Paste live URLs here

