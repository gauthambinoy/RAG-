#!/bin/bash

# ==============================================================================
# RAG System - Complete Setup & Run Script
# ==============================================================================

set -e  # Exit on error

echo "=================================="
echo "RAG System - Complete Setup"
echo "=================================="
echo ""

# Step 1: Check Python
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"
echo ""

# Step 2: Create virtual environment
echo "[2/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi
echo ""

# Step 3: Activate and install dependencies
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Step 4: Check for API key
echo "[4/6] Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ OPENAI_API_KEY not set!"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    echo "Get your key from: https://platform.openai.com/api-keys"
    echo ""
    echo "After setting the key, run this script again."
    exit 1
fi
echo "✓ API key found"
echo ""

# Step 5: Build retrieval index
echo "[5/6] Building retrieval index..."
if [ ! -f "outputs/embeddings/faiss_index.bin" ]; then
    echo "  This will take ~30 seconds (first time only)..."
    python tests/test_retrieval.py > /dev/null 2>&1
    echo "✓ Index built successfully"
else
    echo "✓ Index already exists (skipping)"
fi
echo ""

# Step 6: Test complete system
echo "[6/6] Testing complete RAG system..."
python src/pipeline.py
echo ""

# Final instructions
echo "=================================="
echo "✓ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Your RAG system is ready! You can now:"
echo ""
echo "1. Start the API server:"
echo "   cd deployment"
echo "   uvicorn app:app --reload --port 8000"
echo ""
echo "2. Query via API:"
echo "   curl -X POST http://localhost:8000/query \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"question\": \"What is transformer architecture?\"}'"
echo ""
echo "3. View API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "=================================="
