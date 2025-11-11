# ==============================================================================
# FILE: bm25_retriever.py
# PURPOSE: BM25 sparse retrieval for hybrid search
# ==============================================================================

"""
BM25 (Best Matching 25) - Sparse keyword-based retrieval.

WHY BM25?
- Captures exact term matches that embeddings might miss
- Fast (no GPU needed)
- Complementary to dense semantic search
- Industry standard for hybrid retrieval

HYBRID STRATEGY:
- Dense retrieval: Semantic understanding (synonyms, context)
- BM25: Exact keywords, technical terms, proper nouns
- Fusion: Reciprocal Rank Fusion (RRF) combines both

USAGE:
    from src.retrieval.bm25_retriever import BM25Retriever
    
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    results = bm25.retrieve("transformer architecture", k=10)
"""

import os
import pickle
from typing import List, Dict, Optional
from collections import Counter
import math
import re


# ==============================================================================
# BM25 IMPLEMENTATION
# ==============================================================================

class BM25Retriever:
    """
    BM25 sparse retrieval using TF-IDF weighting.
    
    ALGORITHM:
        BM25(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / 
                     (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))
    
    PARAMETERS:
        k1: Term frequency saturation (default: 1.5)
        b: Length normalization (default: 0.75)
    
    ATTRIBUTES:
        chunks: Document chunks with metadata
        doc_freqs: Document frequencies for IDF
        idf_scores: Precomputed IDF values
        avg_doc_len: Average document length
        k1, b: BM25 parameters
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        PARAMETERS:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.tokenized_docs: List[List[str]] = []
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING BM25 RETRIEVER")
        print(f"{'='*80}")
        print(f"Parameters: k1={k1}, b={b}")
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase + alphanumeric."""
        return re.findall(r'\w+', text.lower())
    
    def build_index(
        self,
        chunks: List[Dict],
        save_to_cache: bool = True,
        cache_path: str = "outputs/embeddings/bm25_index.pkl"
    ):
        """
        Build BM25 index from document chunks.
        
        PARAMETERS:
            chunks: List of document chunks with 'text' field
            save_to_cache: Save index for fast reloading
            cache_path: Path to save index
        
        PROCESS:
            1. Tokenize all documents
            2. Compute document frequencies (DF)
            3. Compute IDF = log((N - DF + 0.5) / (DF + 0.5) + 1)
            4. Store for fast retrieval
        """
        print(f"\n{'='*80}")
        print(f"BUILDING BM25 INDEX")
        print(f"{'='*80}")
        print(f"Number of chunks: {len(chunks)}")
        
        self.chunks = chunks
        
        # Tokenize all documents
        print(f"Tokenizing documents...")
        self.tokenized_docs = [self.tokenize(chunk['text']) for chunk in chunks]
        
        # Compute average document length
        total_len = sum(len(doc) for doc in self.tokenized_docs)
        self.avg_doc_len = total_len / len(self.tokenized_docs) if self.tokenized_docs else 0
        
        # Compute document frequencies
        print(f"Computing document frequencies...")
        self.doc_freqs = {}
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Compute IDF scores
        N = len(self.tokenized_docs)
        self.idf_scores = {}
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf_scores[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        print(f"✓ Index built: {len(self.idf_scores)} unique terms")
        print(f"  Avg doc length: {self.avg_doc_len:.1f} tokens")
        
        # Save to cache
        if save_to_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            index_data = {
                'chunks': self.chunks,
                'tokenized_docs': self.tokenized_docs,
                'doc_freqs': self.doc_freqs,
                'idf_scores': self.idf_scores,
                'avg_doc_len': self.avg_doc_len,
                'k1': self.k1,
                'b': self.b
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(index_data, f)
            print(f"✓ Index saved to: {cache_path}")
        
        print(f"{'='*80}\n")
    
    def load_index(self, cache_path: str = "outputs/embeddings/bm25_index.pkl"):
        """Load pre-built BM25 index from cache."""
        print(f"\n{'='*80}")
        print(f"LOADING BM25 INDEX")
        print(f"{'='*80}")
        
        with open(cache_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.chunks = index_data['chunks']
        self.tokenized_docs = index_data['tokenized_docs']
        self.doc_freqs = index_data['doc_freqs']
        self.idf_scores = index_data['idf_scores']
        self.avg_doc_len = index_data['avg_doc_len']
        self.k1 = index_data.get('k1', self.k1)
        self.b = index_data.get('b', self.b)
        
        print(f"✓ Index loaded: {len(self.idf_scores)} terms, {len(self.chunks)} docs")
        print(f"{'='*80}\n")
    
    def score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a document.
        
        PARAMETERS:
            query_terms: Tokenized query
            doc_idx: Document index
        
        RETURNS:
            BM25 score (higher = more relevant)
        """
        doc = self.tokenized_docs[doc_idx]
        doc_len = len(doc)
        
        # Term frequency counts
        term_freqs = Counter(doc)
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf_scores:
                continue  # Term not in corpus
            
            idf = self.idf_scores[term]
            tf = term_freqs.get(term, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Retrieve top-k documents using BM25.
        
        PARAMETERS:
            query: Query string
            k: Number of results to return
            verbose: Print debug information
        
        RETURNS:
            List of dicts with 'text', 'source', 'score', 'rank'
        """
        if not self.chunks:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"BM25 RETRIEVAL")
            print(f"{'='*80}")
            print(f"Query: {query}")
        
        # Tokenize query
        query_terms = self.tokenize(query)
        
        if verbose:
            print(f"Query terms: {query_terms}")
        
        # Score all documents
        scores = []
        for idx in range(len(self.chunks)):
            score = self.score_document(query_terms, idx)
            if score > 0:  # Only include docs with non-zero score
                scores.append((idx, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for rank, (idx, score) in enumerate(scores[:k], 1):
            chunk = self.chunks[idx].copy()
            chunk['score'] = score
            chunk['rank'] = rank
            results.append(chunk)
        
        if verbose:
            print(f"✓ Retrieved {len(results)} documents")
            for r in results[:3]:
                print(f"  [Rank {r['rank']}] Score: {r['score']:.3f} | {r['text'][:80]}...")
            print(f"{'='*80}\n")
        
        return results


# ==============================================================================
# HYBRID FUSION
# ==============================================================================

def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> List[Dict]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    
    RRF FORMULA:
        score(d) = Σ (weight / (k + rank))
    
    PARAMETERS:
        dense_results: Results from dense (semantic) retrieval
        sparse_results: Results from BM25
        k: Constant (default: 60)
        dense_weight: Weight for dense retrieval (0-1)
        sparse_weight: Weight for sparse retrieval (0-1)
    
    RETURNS:
        Fused results sorted by combined score
    
    RATIONALE:
        - RRF is robust to score magnitude differences
        - Weights allow tuning precision/recall trade-off
        - Dense (0.6) slightly favored for semantic understanding
        - Sparse (0.4) boosts exact matches
    """
    # Build score map
    scores: Dict[int, float] = {}
    chunks_map: Dict[int, Dict] = {}
    
    # Add dense scores
    for rank, result in enumerate(dense_results, 1):
        chunk_id = result['chunk_id']
        rr_score = dense_weight / (k + rank)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rr_score
        chunks_map[chunk_id] = result
    
    # Add sparse scores
    for rank, result in enumerate(sparse_results, 1):
        chunk_id = result['chunk_id']
        rr_score = sparse_weight / (k + rank)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rr_score
        if chunk_id not in chunks_map:
            chunks_map[chunk_id] = result
    
    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Build result list
    fused = []
    for rank, chunk_id in enumerate(sorted_ids, 1):
        chunk = chunks_map[chunk_id].copy()
        chunk['fusion_score'] = scores[chunk_id]
        chunk['rank'] = rank
        fused.append(chunk)
    
    return fused


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    """Test BM25 with mock data."""
    
    print("="*80)
    print("TESTING: bm25_retriever.py")
    print("="*80)
    
    # Mock chunks
    mock_chunks = [
        {
            'chunk_id': 0,
            'text': 'The Transformer uses self-attention mechanisms to process sequences.',
            'source': 'paper.pdf'
        },
        {
            'chunk_id': 1,
            'text': 'Transformers have revolutionized natural language processing.',
            'source': 'paper.pdf'
        },
        {
            'chunk_id': 2,
            'text': 'The EU AI Act regulates artificial intelligence systems.',
            'source': 'act.docx'
        },
    ]
    
    print("\n" + "-"*80)
    print("TEST 1: Build BM25 Index")
    print("-"*80)
    
    bm25 = BM25Retriever()
    bm25.build_index(mock_chunks, save_to_cache=False)
    
    print("\n" + "-"*80)
    print("TEST 2: Retrieve with BM25")
    print("-"*80)
    
    query = "transformer attention mechanism"
    results = bm25.retrieve(query, k=2, verbose=True)
    
    print("\n" + "="*80)
    print("✓ BM25 TESTS PASSED")
    print("="*80)
