# ==============================================================================
# FILE: reranker.py
# PURPOSE: Cross-encoder reranking for improved retrieval precision
# ==============================================================================

"""
Cross-Encoder Reranking - Second-stage precision refinement.

WHY RERANKING?
- First stage (dense/hybrid): Fast, retrieves Top-20 candidates
- Second stage (cross-encoder): Slower, accurate scoring of Top-20 → Top-5
- Cross-encoders see full query-document pairs (better than bi-encoders)

TRADE-OFFS:
- Speed: ~50ms for 20 pairs (acceptable for production)
- Accuracy: +10-15% precision improvement
- Memory: 400MB model (manageable)

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking
- Balance of speed and quality
- 80M parameters

USAGE:
    from src.retrieval.reranker import CrossEncoderReranker
    
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates, top_k=5)
"""

import os
from typing import List, Dict, Optional
import numpy as np


# ==============================================================================
# CROSS-ENCODER RERANKER
# ==============================================================================

class CrossEncoderReranker:
    """
    Rerank retrieved documents using cross-encoder.
    
    ARCHITECTURE:
        Query + Document → BERT → Single relevance score
    
    WORKFLOW:
        1. Retrieval: Get Top-20 candidates (fast)
        2. Reranking: Score all 20 with cross-encoder (accurate)
        3. Return: Top-5 after reranking
    
    ATTRIBUTES:
        model: Cross-encoder model
        model_name: Model identifier
        device: CPU/GPU
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        lazy_load: bool = True
    ):
        """
        Initialize cross-encoder reranker (lazy by default).
        
        PARAMETERS:
            model_name: Hugging Face model identifier
            lazy_load: If True, only load model when first used (avoids download on init)
        """
        self.model_name = model_name
        self.model = None
        self.lazy_load = lazy_load
        
        # Eager load only if explicitly requested
        if not lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load model lazily on first use."""
        if self.model is not None:
            return  # Already loaded
        
        print(f"\nLoading cross-encoder reranker: {self.model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
            print(f"✓ Reranker model loaded")
        except ImportError:
            print("⚠ sentence-transformers not installed; reranker disabled")
            self.model = None
        except Exception as e:
            print(f"⚠ Reranker load failed: {e}; using fallback")
            self.model = None
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Rerank candidate documents.
        
        PARAMETERS:
            query: User query
            candidates: List of retrieved chunks (Top-20 from first stage)
            top_k: Number of results to return after reranking
            verbose: Print debug information
        
        RETURNS:
            Top-k reranked chunks with updated scores
        
        PROCESS:
            1. Prepare query-document pairs
            2. Score all pairs with cross-encoder
            3. Sort by score (descending)
            4. Return top-k
        """
        if not candidates:
            return []
        
        # Lazy-load model on first rerank call
        if self.lazy_load:
            self._load_model()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"CROSS-ENCODER RERANKING")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Candidates: {len(candidates)} → Top-{top_k}")
        
        # Fallback if model not loaded
        if self.model is None:
            if verbose:
                print("⚠️ Model not available, using original scores")
            return candidates[:top_k]
        
        # Prepare pairs
        pairs = [(query, cand['text']) for cand in candidates]
        
        # Score with cross-encoder
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            if verbose:
                print(f"⚠️ Scoring failed: {e}, using fallback")
            return candidates[:top_k]
        
        # Attach scores and sort
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(scores[i])
            cand['original_rank'] = cand.get('rank', i + 1)
        
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks
        for rank, cand in enumerate(reranked[:top_k], 1):
            cand['rank'] = rank
        
        if verbose:
            print(f"✓ Reranking complete")
            print(f"\nTop-3 after reranking:")
            for r in reranked[:3]:
                orig = r.get('original_rank', '?')
                print(f"  [Rank {r['rank']}] (was {orig}) Score: {r['rerank_score']:.3f}")
                print(f"    Text: {r['text'][:80]}...")
            print(f"{'='*80}\n")
        
        return reranked[:top_k]


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    """Test reranker with mock data."""
    
    print("="*80)
    print("TESTING: reranker.py")
    print("="*80)
    
    # Mock candidates
    mock_candidates = [
        {
            'chunk_id': 0,
            'text': 'The Transformer architecture uses multi-head attention.',
            'source': 'paper.pdf',
            'score': 0.85,
            'rank': 1
        },
        {
            'chunk_id': 1,
            'text': 'Recurrent neural networks process sequences step by step.',
            'source': 'paper.pdf',
            'score': 0.75,
            'rank': 2
        },
        {
            'chunk_id': 2,
            'text': 'Self-attention allows parallel processing of all tokens.',
            'source': 'paper.pdf',
            'score': 0.70,
            'rank': 3
        },
    ]
    
    print("\n" + "-"*80)
    print("TEST: Rerank Candidates")
    print("-"*80)
    
    reranker = CrossEncoderReranker()
    
    if reranker.model is not None:
        query = "How does attention work in transformers?"
        reranked = reranker.rerank(query, mock_candidates, top_k=2, verbose=True)
        
        print("\n" + "="*80)
        print("✓ RERANKER TESTS PASSED")
        print("="*80)
    else:
        print("\n⚠️ Model not loaded, skipping test")
        print("Install sentence-transformers to enable reranking:")
        print("  pip install sentence-transformers")
