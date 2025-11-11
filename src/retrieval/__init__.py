# Retrieval Package

"""
Retrieval module for RAG system.

INCLUDES:
- Base retriever
- Optimized V2 (query-type-aware ranking)
"""

from .retriever_optimized_v2 import RetrieverOptimizedV2, QueryTopicDetector

__all__ = [
    'RetrieverOptimizedV2',
    'QueryTopicDetector'
]