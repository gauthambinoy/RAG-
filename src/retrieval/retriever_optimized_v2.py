# ==============================================================================
# FILE: retriever_optimized_v2.py
# PURPOSE: Enhanced retriever with document-type-aware ranking
# ==============================================================================

"""
RETRIEVER OPTIMIZED V2 - Query-Type Aware Ranking

This module enhances the base retriever with intelligent query routing
and document-type boosting.

NEW FEATURES:
1. Query Topic Detection - Classify queries (technical/legal/data/mixed)
2. Type-Aware Boosting - Boost scores from relevant document types by 15%
3. Adaptive Ranking - Re-rank results based on query-type affinity
4. Performance Metrics - Track query type distribution and hit rates

ALGORITHM:
1. Detect query topic
2. Map to relevant document types
3. Retrieve candidates (retrieve_dense or retrieve_hybrid)
4. BOOST scores from matching document types
5. Re-rank and return top-k

EXPECTED IMPROVEMENTS:
- Technical queries: +2-5% accuracy
- Legal queries: +3-7% accuracy
- Data queries: +10-30% accuracy
- Mixed queries: +5-10% accuracy
- Overall: +8-15% improvement
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==============================================================================
# QUERY TOPIC DETECTOR
# ==============================================================================

class QueryTopicDetector:
    """
    Detect query topic to enable intelligent routing.
    
    Topics:
    - TECHNICAL: About transformers, neural networks, AI models
    - LEGAL: About AI Act, regulations, compliance
    - DATA: About inflation, rates, numerical trends
    - MIXED: Contains elements from multiple topics
    - GENERAL: Cannot determine
    """
    
    # Keywords per topic
    TECHNICAL_KEYWORDS = {
        'transformer', 'attention', 'neural', 'model', 'deepseek', 'r1',
        'training', 'algorithm', 'embedding', 'layer', 'token', 'architecture',
        'reinforcement', 'learning', 'optimization', 'gradient', 'loss'
    }
    
    LEGAL_KEYWORDS = {
        'ai act', 'regulation', 'compliance', 'penalty', 'prohibited',
        'article', 'requirement', 'high-risk', 'transparency', 'provider',
        'operator', 'user', 'fine', 'shall', 'may not'
    }
    
    DATA_KEYWORDS = {
        'inflation', 'rate', 'year', 'trend', 'index', 'value', 'percentage',
        'data', 'calculator', 'quarter', 'month', 'historical', 'time-series'
    }
    
    @staticmethod
    def detect_topic(query: str) -> Tuple[str, float]:
        """
        Detect query topic.
        
        PARAMETERS:
            query (str): User query
            
        RETURNS:
            Tuple[str, float]: (topic, confidence)
                topic: 'technical', 'legal', 'data', 'mixed', 'general'
                confidence: 0.0-1.0 score
        """
        
        query_lower = query.lower()
        
        # Count keyword matches
        technical_count = sum(1 for kw in QueryTopicDetector.TECHNICAL_KEYWORDS 
                            if kw in query_lower)
        legal_count = sum(1 for kw in QueryTopicDetector.LEGAL_KEYWORDS 
                         if kw in query_lower)
        data_count = sum(1 for kw in QueryTopicDetector.DATA_KEYWORDS 
                        if kw in query_lower)
        
        # Determine topic
        scores = {
            'technical': technical_count,
            'legal': legal_count,
            'data': data_count
        }
        
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]
        
        # Check for mixed topics
        topics_with_keywords = sum(1 for score in scores.values() if score > 0)
        
        if topics_with_keywords > 1:
            topic = 'mixed'
            confidence = best_score / (best_score + 2)  # Penalize mixed
        elif best_score > 0:
            topic = best_topic
            confidence = min(0.95, 0.6 + best_score * 0.1)
        else:
            topic = 'general'
            confidence = 0.3
        
        return topic, confidence


# ==============================================================================
# RETRIEVER OPTIMIZED V2
# ==============================================================================

class RetrieverOptimizedV2:
    """
    Enhanced retriever with query-type-aware ranking.
    
    Builds on RetrieverOptimized by adding:
    1. Query topic detection
    2. Document type boosting
    3. Type-aware re-ranking
    4. Performance tracking
    """
    
    # Mapping of query topics to relevant document types
    TOPIC_TO_DOC_TYPES = {
        'technical': {'technical_paper'},
        'legal': {'policy_legal'},
        'data': {'tabular_data'},
        'mixed': {'technical_paper', 'policy_legal', 'tabular_data'},
        'general': {'technical_paper', 'policy_legal', 'tabular_data'}
    }
    
    # Boost factors per document type match
    BOOST_FACTOR = 1.15  # 15% boost for matching type
    
    def __init__(self, base_retriever):
        """
        Initialize V2 retriever.
        
        PARAMETERS:
            base_retriever: RetrieverOptimized instance
        """
        self.base_retriever = base_retriever
        self.detector = QueryTopicDetector()
        
        # Metrics
        self.metrics = {
            'total_queries': 0,
            'topic_distribution': defaultdict(int),
            'type_matches': 0,
            'boost_applied': 0
        }
        
        logger.info("✅ RetrieverOptimizedV2 initialized with type-aware ranking")
    
    def retrieve_with_type_awareness(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid"
    ) -> List[Dict]:
        """
        Retrieve with intelligent query-type awareness.
        
        WORKFLOW:
        1. Detect query topic
        2. Determine relevant document types
        3. Retrieve candidates using base retriever
        4. Boost scores for matching document types
        5. Re-rank and return top-k
        
        PARAMETERS:
            query (str): User query
            k (int): Number of results to return
            method (str): Retrieval method
            
        RETURNS:
            List[Dict]: Top-k results with boosted scores
        """
        
        # Step 1: Detect query topic
        topic, confidence = self.detector.detect_topic(query)
        self.metrics['total_queries'] += 1
        self.metrics['topic_distribution'][topic] += 1
        
        logger.info(f"\n📊 QUERY ANALYSIS")
        logger.info(f"   Topic: {topic} (confidence: {confidence:.2f})")
        logger.info(f"   Query: {query[:60]}...")
        
        # Step 2: Determine relevant document types
        relevant_types = self.TOPIC_TO_DOC_TYPES.get(topic, set())
        logger.info(f"   Relevant doc types: {', '.join(relevant_types) if relevant_types else 'all'}")
        
        # Step 3: Retrieve candidates (get more to allow for re-ranking)
        retrieval_k = min(k * 3, 20)  # Get 3x candidates for better re-ranking
        
        try:
            candidates = self.base_retriever.retrieve(query, k=retrieval_k, method=method)
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
        
        logger.info(f"   Retrieved {len(candidates)} candidates")
        
        # Step 4: Boost scores for matching document types
        boosted_results = []
        types_found = set()
        
        for result in candidates:
            doc_type = result.get('metadata', {}).get('document_type', 'unknown')
            types_found.add(doc_type)
            
            # Apply boost if document type matches query topic
            if doc_type in relevant_types:
                original_score = result.get('score', 0)
                result['score'] *= self.BOOST_FACTOR
                result['boost_applied'] = True
                result['original_score'] = original_score
                self.metrics['boost_applied'] += 1
                logger.info(f"   ✅ Boost applied to {doc_type} chunk")
            else:
                result['boost_applied'] = False
            
            boosted_results.append(result)
        
        self.metrics['type_matches'] += len([r for r in boosted_results if r.get('boost_applied')])
        
        # Step 5: Re-rank by boosted score and return top-k
        final_results = sorted(
            boosted_results,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:k]
        
        # Log results
        logger.info(f"\n✅ FINAL RESULTS (Top {len(final_results)})")
        for i, result in enumerate(final_results):
            doc_type = result.get('metadata', {}).get('document_type', '?')
            score = result.get('score', 0)
            boosted = "🔥" if result.get('boost_applied') else "  "
            logger.info(f"   {i+1}. [{doc_type}] {boosted} Score: {score:.4f}")
        
        return final_results
    
    def retrieve(self, query: str, k: int = 5, method: str = "auto") -> List[Dict]:
        """
        Main retrieve method - delegates to type-aware retrieval.
        
        PARAMETERS:
            query (str): User query
            k (int): Number of results
            method (str): Retrieval method
            
        RETURNS:
            List[Dict]: Retrieved chunks
        """
        return self.retrieve_with_type_awareness(query, k=k, method=method)
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        RETURNS:
            Dict: Metrics including topic distribution and boost stats
        """
        
        metrics = {
            "total_queries": self.metrics['total_queries'],
            "topic_distribution": dict(self.metrics['topic_distribution']),
            "type_matches": self.metrics['type_matches'],
            "boosts_applied": self.metrics['boost_applied'],
            "boost_hit_rate": (
                self.metrics['boost_applied'] / self.metrics['total_queries']
                if self.metrics['total_queries'] > 0
                else 0
            )
        }
        
        # Also get base retriever metrics
        try:
            base_metrics = self.base_retriever.get_metrics()
            metrics['base_retriever'] = base_metrics
        except:
            pass
        
        return metrics
    
    def print_metrics(self):
        """Pretty print metrics"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RETRIEVER V2 METRICS")
        logger.info(f"{'='*70}")
        
        metrics = self.get_metrics()
        
        logger.info(f"Total Queries: {metrics['total_queries']}")
        logger.info(f"Topic Distribution:")
        for topic, count in metrics['topic_distribution'].items():
            pct = 100 * count / max(metrics['total_queries'], 1)
            logger.info(f"  - {topic}: {count} ({pct:.1f}%)")
        
        logger.info(f"\nType-Aware Boosting:")
        logger.info(f"  - Boosts Applied: {metrics['boosts_applied']}")
        logger.info(f"  - Hit Rate: {metrics['boost_hit_rate']:.1%}")
        
        if 'base_retriever' in metrics:
            logger.info(f"\nBase Retriever:")
            for key, value in metrics['base_retriever'].items():
                logger.info(f"  - {key}: {value}")
        
        logger.info(f"{'='*70}\n")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    """Test query topic detection"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    detector = QueryTopicDetector()
    
    test_queries = [
        # Technical
        "How does the transformer attention mechanism work?",
        "What is DeepSeek-R1 and how does it compare to other models?",
        "Explain the architecture of neural networks",
        
        # Legal
        "What are the high-risk AI systems under the EU AI Act?",
        "What are the penalties for non-compliance with AI regulations?",
        "Explain the transparency requirements for AI providers",
        
        # Data
        "What was the inflation rate in 2023?",
        "How has the price index changed over time?",
        "Calculate inflation for the past 5 years",
        
        # Mixed
        "How do transformers work and what are the legal implications?",
        "Explain attention mechanisms and their compliance requirements",
        
        # General
        "What is this about?",
        "Tell me something interesting"
    ]
    
    print("\n" + "="*70)
    print("QUERY TOPIC DETECTION TEST")
    print("="*70)
    
    for query in test_queries:
        topic, confidence = detector.detect_topic(query)
        print(f"\nQuery: {query}")
        print(f"  Topic: {topic} (confidence: {confidence:.2f})")
