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

Fully managed HTTPS + autoscaling. Integrated CI here pushes images to ECR and can deploy App Runner via CloudFormation.

Prereqs
- AWS account ID (you provided: `920822856790`)
- Region (we use: `eu-west-1`)
- GitHub OIDC IAM role with ECR + CloudFormation + App Runner permissions; store its ARN in repository variable `AWS_ROLE_TO_ASSUME`
- GitHub secret `GEMINI_API_KEY`

### 1. Image Build & Push (Automatic on each push to `main`)
Workflow: `.github/workflows/ecr-push.yml` (already created)
Action: When you push to `main`, it:
1. Assumes `AWS_ROLE_TO_ASSUME` via OIDC
2. Builds Docker image
3. Tags `rag:main-<commit-sha>`
4. Pushes to `920822856790.dkr.ecr.eu-west-1.amazonaws.com/rag`

Manual trigger alternative if needed:
```bash
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=920822856790
export ECR_REPO=rag
IMAGE_TAG=manual-$(date +%Y%m%d%H%M)
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" || true
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com
docker build -t "$ECR_REPO:$IMAGE_TAG" .
docker tag "$ECR_REPO:$IMAGE_TAG" "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO:$IMAGE_TAG"
docker push "$AWS_ACCOUNT_ID".dkr.ecr."$AWS_REGION".amazonaws.com/"$ECR_REPO:$IMAGE_TAG"
```

### 2. Deploy / Update App Runner Service
Workflow: `.github/workflows/deploy-apprunner.yml` (manual dispatch)
Steps:
1. In GitHub → Actions → Deploy App Runner → Run workflow
2. (Optional) Provide `image_tag` (else it finds the latest pushed)
3. Workflow deploys CloudFormation template `aws/app-runner-service.yaml` creating/updating:
   - IAM role granting App Runner pull access to ECR
   - App Runner service with env vars (`GEMINI_API_KEY`, `MODEL_NAME`, `LOG_LEVEL`, `RAG_ENABLE_RERANKER`)
4. Outputs the public service URL.

### 3. Verify
```bash
curl -s https://<service-url>/health
curl -s https://<service-url>/docs
```

### 4. Custom Domain (Optional)
1. Request ACM certificate in `eu-west-1` for `yourdomain.com` + `api.yourdomain.com`
2. Validate via Route 53 CNAMEs
3. App Runner → Custom domains → Add domain → Attach certificate → Add CNAME record pointing to target
4. Wait status `Active`; re-test `/health`.

### 5. Required Permissions Summary for OIDC Role
Attach policy with:
- `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`
- `ecr:DescribeImages`
- `cloudformation:CreateStack`, `cloudformation:UpdateStack`, `cloudformation:DescribeStacks`
- `apprunner:CreateService`, `apprunner:UpdateService`, `apprunner:DescribeService`, `apprunner:ListServices`
- `iam:PassRole` (scoped to the created App Runner ECR access role ARN pattern)

### 6. After First Deploy
Update `SUBMISSION.md` live URLs:
- API: `<service-url>/docs`
- Health: `<service-url>/health`

### Troubleshooting
| Symptom | Fix |
|---------|-----|
| ECR image not found | Ensure ecr-push workflow ran and repository `rag` exists. |
| CloudFormation failure on IAM | Check OIDC role has `CAPABILITY_NAMED_IAM` permission and `iam:PassRole`. |
| 502 errors | Confirm health path `/health` reachable locally and service port set to 8000. |
| Missing Gemini key | Set `GEMINI_API_KEY` secret in GitHub before dispatching deploy workflow. |

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
