# ==============================================================================
# FILE: metrics.py
# PURPOSE: Evaluation metrics for RAG (retrieval + generation)
# ==============================================================================

from typing import List, Dict, Tuple, Optional
import os
import re

# ----------------------------- Retrieval Metrics -----------------------------

def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """Compute Precision@K using unique source filenames as relevance proxy.
    Deduplicates retrieved sources to avoid inflation when multiple chunks share
    the same source file.
    """
    if k <= 0:
        return 0.0
    # Deduplicate preserving order
    dedup: List[str] = []
    seen = set()
    for s in retrieved_sources:
        sl = s.lower()
        if sl not in seen:
            dedup.append(s)
            seen.add(sl)
    topk = dedup[:k]
    rel_set = set(s.lower() for s in relevant_sources)
    hits = sum(1 for s in topk if s.lower() in rel_set)
    return hits / max(1, len(topk))


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """Compute Recall@K using unique sources as relevance proxy."""
    rel_set = set(s.lower() for s in relevant_sources)
    if not rel_set:
        return 0.0
    dedup: List[str] = []
    seen = set()
    for s in retrieved_sources:
        sl = s.lower()
        if sl not in seen:
            dedup.append(s)
            seen.add(sl)
    topk = set(s.lower() for s in dedup[:k])
    hits = len(rel_set.intersection(topk))
    return hits / len(rel_set)


def mean_reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """Compute MRR using rank of first relevant unique source."""
    rel_set = set(s.lower() for s in relevant_sources)
    seen = set()
    rank = 0
    for src in retrieved_sources:
        sl = src.lower()
        if sl in seen:
            continue
        seen.add(sl)
        rank += 1
        if sl in rel_set:
            return 1.0 / rank
    return 0.0


# ---------------------------- Generation Metrics -----------------------------

def contains_expected_phrases(answer: str, expected_phrases: List[str]) -> float:
    """Return fraction of expected phrases found in the answer (case-insensitive)."""
    if not expected_phrases:
        return 0.0
    ans = answer.lower()
    hits = sum(1 for p in expected_phrases if p.lower() in ans)
    return hits / len(expected_phrases)


def simple_hallucination_flag(answer: str, context: str) -> bool:
    """Improved heuristic to reduce false positives.

    Rules (short-circuit in order):
    - Empty or very short answers (< 25 tokens): not hallucination.
    - If answer contains the explicit refusal phrase ("I cannot answer this based on the provided documents"), not hallucination.
    - Sentence-level containment: if >= 1 sentence (>= 8 tokens) is 60%+ contained in context, not hallucination.
    - Content-word Jaccard: compute Jaccard over non-stopword tokens; if >= 0.22 (lowered threshold), not hallucination.
    - Otherwise, flag as hallucination.
    """
    ans = answer.strip()
    if not ans:
        return False
    answer_tokens = re.findall(r"\w+", ans.lower())
    if len(answer_tokens) < 25:
        return False
    if "i cannot answer this based on the provided documents" in ans.lower():
        return False

    ctx = context.lower()
    # Sentence-level check
    sentences = re.split(r"[.!?]\s+", ans)
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        if len(toks) >= 8:
            # crude containment via substring
            if s.lower()[:200] and s.lower()[:200] in ctx:
                return False

    # Content-word Jaccard (remove common stopwords) - lowered threshold from 0.28 to 0.22
    stop = {
        'the','is','a','an','and','or','to','of','in','on','for','by','with','as','that','this','it','be','are','was','were','from','at','we','you','they','he','she','but','not','than','which','their','its','our','your','i'
    }
    ans_set = {t for t in answer_tokens if t not in stop}
    ctx_tokens = re.findall(r"\w+", ctx)
    ctx_set = {t for t in ctx_tokens if t not in stop}
    if not ans_set or not ctx_set:
        return False
    inter = len(ans_set & ctx_set)
    union = len(ans_set | ctx_set)
    jacc = inter / union if union else 0.0
    return jacc < 0.22  # Lowered from 0.28 to reduce false positives


def citation_coverage(answer: str) -> float:
    """Compute citation coverage: fraction of sentences with [C#] citations.
    
    Returns value between 0 and 1:
    - 1.0: All sentences have citations
    - 0.0: No citations at all
    """
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  # Filter very short
    
    if not sentences:
        return 0.0
    
    # Count sentences with [C#] pattern
    cited = sum(1 for s in sentences if re.search(r'\[C\d+\]', s))
    
    return cited / len(sentences)


# ------------------------------ Utility Helpers ------------------------------

def extract_sources_from_chunks(chunks: List[Dict]) -> List[str]:
    """Return normalized source names (basenames only) for fair comparison.
    Chunks may store full relative paths like 'data/pdfs/file.pdf';
    expected sources in tests often use just 'file.pdf'.
    """
    sources: List[str] = []
    for c in chunks:
        src = c.get('source', 'unknown') or 'unknown'
        sources.append(os.path.basename(src))
    return sources
