"""Build retrieval indices (FAISS + BM25) from documents under data/.

Usage examples:
  # Fast, BM25-only (no embedding downloads)
  RAG_DISABLE_EMBEDDINGS=1 python scripts/build_index.py

  # Build dense + BM25 (requires embedding model available)
  python scripts/build_index.py --dense

This script will:
  - Load documents from data/pdfs, data/documents, data/tables
  - Normalize and chunk text
  - Build indices via Retriever (saves under outputs/embeddings)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict

from src.preprocessing.chunker import normalize_text, chunk_text
from src.loaders.pdf_loader import load_pdf_data
from src.loaders.docx_loader import load_docx_data
from src.loaders.excel_loader import load_excel_or_csv
from src.retrieval.retriever import Retriever


def collect_documents() -> List[Dict]:
    base = Path('data')
    pdf_dir = base / 'pdfs'
    doc_dir = base / 'documents'
    tbl_dir = base / 'tables'

    files: List[Path] = []
    for d in [pdf_dir, doc_dir, tbl_dir]:
        if d.exists():
            for p in d.rglob('*'):
                if p.suffix.lower() in {'.pdf', '.docx', '.xlsx', '.xls', '.csv'}:
                    files.append(p)

    docs: List[Dict] = []
    for path in files:
        text = None
        if path.suffix.lower() == '.pdf':
            text = load_pdf_data(str(path))
        elif path.suffix.lower() == '.docx':
            text = load_docx_data(str(path))
        elif path.suffix.lower() in {'.xlsx', '.xls', '.csv'}:
            text = load_excel_or_csv(str(path))

        if text:
            docs.append({
                'source': str(path),
                'text': text
            })

    return docs


def build_chunks(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    chunks: List[Dict] = []
    cid = 0
    for d in docs:
        norm = normalize_text(d['text'])
        for ch in chunk_text(norm, chunk_size=chunk_size, overlap=overlap):
            chunks.append({
                'chunk_id': cid,
                'text': ch['text'],
                'source': d['source']
            })
            cid += 1
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk-size', type=int, default=800)
    ap.add_argument('--overlap', type=int, default=100)
    ap.add_argument('--dense', action='store_true', help='Enable dense embeddings (default off if RAG_DISABLE_EMBEDDINGS=1).')
    args = ap.parse_args()

    if args.dense:
        # Ensure embeddings are enabled unless user explicitly disables
        os.environ.pop('RAG_DISABLE_EMBEDDINGS', None)
    else:
        # Default to BM25-only unless user passes --dense
        os.environ['RAG_DISABLE_EMBEDDINGS'] = '1'

    print("Collecting documents from data/ ...")
    docs = collect_documents()
    if not docs:
        print("No documents found under data/. Nothing to index.")
        return
    print(f"Found {len(docs)} documents")

    print("Building chunks ...")
    chunks = build_chunks(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Built {len(chunks)} chunks")

    # Initialize retriever (hybrid to ensure BM25 index is built; dense controlled by env)
    retriever = Retriever(use_hybrid=True, use_reranker=False, lazy_embedding=True)
    retriever.build_index(chunks, use_cache=True)

    print("Index build complete. Files saved under outputs/embeddings/")


if __name__ == '__main__':
    main()
