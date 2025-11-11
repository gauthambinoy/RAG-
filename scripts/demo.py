#!/usr/bin/env python3
"""
Simple demo script to test the RAG system with example queries.
Shows: Answer, sources, provider, model, and response time.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RAGPipeline

# Example queries to test
EXAMPLE_QUERIES = [
    "What is the transformer architecture?",
    "What are the main provisions of the EU AI Act?",
    "What was the inflation rate in 2020?",
    "What is DeepSeek-R1 and how does it use reinforcement learning?",
]

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    pipeline = RAGPipeline(verbose_init=False)
    pipeline.load_index()

    for query in EXAMPLE_QUERIES:
        start = time.time()
        result = pipeline.query(query)
        dt = time.time() - start
        answer = (result.get('answer','')[:180] + '…') if len(result.get('answer','')) > 180 else result.get('answer','')
        sources = ", ".join(dict.fromkeys(result.get('sources', [])))
        print(f"Q: {query}\nA: {answer}\nSources: {sources or 'None'}\nLatency: {dt:.2f}s | Model: {result.get('model','N/A')}\n---")

if __name__ == "__main__":
    main()
