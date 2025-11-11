"""Run all test queries and print full untruncated results.

Usage (BM25-only, fastest):
    RAG_DISABLE_EMBEDDINGS=1 python scripts/run_all_queries.py

Usage (hybrid with embeddings if already cached):
    python scripts/run_all_queries.py

This prints for each query:
    [ID] <question>
    Latency: <seconds>
    Sources: <list>
    Answer:
    <full answer>
    ---
"""

import os
import time
from pathlib import Path

from src.pipeline import RAGPipeline
from src.queries.test_queries import get_all_queries


def main():
    disable = os.getenv("RAG_DISABLE_EMBEDDINGS", "0") == "1"
    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"all_queries_full_{int(time.time())}.txt"
    print("=" * 100)
    print("RUNNING ALL TEST QUERIES")
    print(f"Mode: {'BM25-only' if disable else 'Hybrid (if embeddings available)'}")
    print("=" * 100)

    pipeline = RAGPipeline(verbose_init=False)
    try:
        pipeline.load_index()
    except Exception as e:
        print(f"⚠ Could not load index: {e}")
        print("You may need to build the index first.")

    queries = get_all_queries()
    print(f"Total queries: {len(queries)}\n")

    blocks = []

    for q in queries:
        qid = q['id']
        text = q['query']
        print(f"[{qid}] {text}")
        start = time.time()
        try:
            result = pipeline.query(text, top_k=5, verbose=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            print("-" * 100)
            continue
        latency = time.time() - start
        chunks = result.get('retrieved_chunks', [])
        sources = [Path(c.get('source') or c.get('file_path','Unknown')).name for c in chunks]
        # Deduplicate while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)
        block = []
        block.append(f"[{qid}] {text}")
        block.append(f"Latency: {latency:.2f}s")
        block.append("Sources: " + (", ".join(unique_sources) or "None"))
        block.append("Answer:")
        block.append(result.get('answer', '').strip() or "<no answer>")
        block.append("-" * 100)
        blocks.append("\n".join(block))

        # Also print to terminal
        print(blocks[-1])

    # Write full, untruncated output to file
    outfile.write_text("\n".join(blocks), encoding="utf-8")
    print("DONE.")
    print(f"Saved full results to: {outfile}")


if __name__ == "__main__":
    main()
