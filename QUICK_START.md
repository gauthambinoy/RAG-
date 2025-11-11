# 🚀 Quick Start Guide - Run the RAG System

## Three Ways to See the System in Action

### ✨ **Option 1: Interactive Web UI (Streamlit)** - RECOMMENDED

This gives you a beautiful interactive interface to ask questions and see results in real-time.

```bash
# Already running! Open in your browser:
# 🌐 http://localhost:8502

# Or run it yourself:
cd /home/gautham/Desktop/DATA\ SCIENTIST/DATA\ 2/RAG
source .venv/bin/activate
streamlit run deployment/streamlit_app.py
```

**What you'll see:**
- ✅ Question input box
- ✅ Real-time answer generation
- ✅ Retrieved source documents with scores
- ✅ Citations highlighted in the answer
- ✅ Response time and provider information
- ✅ Query history

**Try these example queries:**
1. "What is the transformer architecture?"
2. "What are the main provisions of the EU AI Act?"
3. "What was the inflation rate in 2020?"

---

### 🔧 **Option 2: REST API (FastAPI)**

This gives you a REST API to integrate into other applications.

```bash
# Terminal 1: Start the API server
cd /home/gautham/Desktop/DATA\ SCIENTIST/DATA\ 2/RAG
source .venv/bin/activate
cd deployment
python app.py

# Server will start at: http://localhost:8000
```

**Test the API:**

```bash
# Terminal 2: Test health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer architecture?",
    "k": 5
  }'

# Check system stats
curl http://localhost:8000/stats
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /query` - Ask a question (body: `{"query": "...", "k": 5}`)

---

### 📊 **Option 3: View Evaluation Results**

See comprehensive evaluation metrics from the automated tests.

```bash
# View the evaluation report
cd /home/gautham/Desktop/DATA\ SCIENTIST/DATA\ 2/RAG
cat outputs/evaluations/metrics_report.md

# Or open in browser
xdg-open outputs/evaluations/metrics_report.md
```

**What you'll see:**
```
## Aggregate Metrics
- avg_precision_at_5: 1.0000
- avg_recall_at_5: 0.9500
- avg_mrr: 1.0000
- avg_phrase_coverage: 0.5700
- avg_citation_coverage: 0.4922
- hallucination_rate: 0.3000
- num_queries: 10

## Per-Query Results
[Detailed breakdown for each test query]
```

**Run fresh evaluation:**
```bash
cd /home/gautham/Desktop/DATA\ SCIENTIST/DATA\ 2/RAG
source .venv/bin/activate
python scripts/run_evaluation.py

# Results saved to:
# - outputs/evaluations/metrics_report.json
# - outputs/evaluations/metrics_report.md
```

---

## 🎯 Quick Demo Script

Copy and paste this entire block to see everything:

```bash
# Navigate to project
cd /home/gautham/Desktop/DATA\ SCIENTIST/DATA\ 2/RAG

# Activate environment
source .venv/bin/activate

# 1. View evaluation results
echo "📊 EVALUATION RESULTS:"
echo "====================="
head -20 outputs/evaluations/metrics_report.md
echo ""

# 2. Test simple query via Python
echo "🔍 TESTING SIMPLE QUERY:"
echo "======================="
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/gautham/Desktop/DATA SCIENTIST/DATA 2/RAG')

from src.pipeline import RAGPipeline

# Load pipeline
print("Loading pipeline...")
pipeline = RAGPipeline()
pipeline.load_index()

# Query
print("\nQuerying: 'What is the transformer architecture?'\n")
result = pipeline.query("What is the transformer architecture?", k=3)

print(f"ANSWER:\n{result['answer'][:500]}...\n")
print(f"SOURCES: {', '.join(result['sources'])}")
print(f"PROVIDER: {result.get('provider', 'unknown')}")
print(f"MODEL: {result.get('model', 'unknown')}")
EOF

echo ""
echo "✅ Done! Now open Streamlit UI at: http://localhost:8502"
```

---

## 📁 Where to Find Outputs

### Evaluation Reports
```bash
# Markdown report (human-readable)
outputs/evaluations/metrics_report.md

# JSON report (machine-readable)
outputs/evaluations/metrics_report.json
```

### Cached Indexes
```bash
# FAISS dense vector index
outputs/embeddings/faiss_index.bin

# BM25 sparse index
outputs/embeddings/bm25_index.pkl

# Chunk metadata
outputs/embeddings/chunks_metadata.pkl
```

### Logs
```bash
# System logs (if enabled)
outputs/logs/
```

---

## 🌐 Currently Running

**Streamlit UI is already running!**
- 🌐 **Local:** http://localhost:8502
- 🌐 **Network:** http://172.20.10.10:8502
- 🌐 **External:** http://212.129.79.123:8502

**Just open your browser and go to one of these URLs!**

---

## ❓ Common Commands

### View Evaluation Results
```bash
cat outputs/evaluations/metrics_report.md | less
```

### Run New Evaluation
```bash
python scripts/run_evaluation.py
```

### Start Streamlit UI
```bash
streamlit run deployment/streamlit_app.py
```

### Start FastAPI Server
```bash
cd deployment && python app.py
```

### Run Tests
```bash
pytest tests/test_retrieval.py -v
```

### Check System Stats
```bash
python3 << 'EOF'
from src.pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.load_index()
print(pipeline.get_stats())
EOF
```

---

## 🎨 What Each Interface Shows

### Streamlit UI (http://localhost:8502)
- ✅ Interactive question input
- ✅ Real-time answer generation
- ✅ Retrieved contexts with scores
- ✅ Citation highlighting
- ✅ Provider and model info
- ✅ Response time tracking
- ✅ Query history

### FastAPI API (http://localhost:8000)
- ✅ REST endpoints for integration
- ✅ JSON request/response
- ✅ OpenAPI documentation at /docs
- ✅ Health checks
- ✅ System statistics

### Evaluation Reports
- ✅ Precision, Recall, MRR metrics
- ✅ Phrase coverage analysis
- ✅ Citation coverage tracking
- ✅ Hallucination detection
- ✅ Per-query breakdown

---

## 🚀 Next Steps

1. **Open the UI:** http://localhost:8502
2. **Ask a question** from the examples
3. **See the results** with citations and sources
4. **Check evaluation metrics** in `outputs/evaluations/metrics_report.md`
5. **Read full docs** in `docs/COMPLETE_IMPLEMENTATION_SUMMARY.md`

---

**You're all set! The system is running and ready to use! 🎉**
