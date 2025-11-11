"""
Answer Caching System
=====================

Implements caching for RAG queries to improve response time and reduce API costs.

BENEFITS:
- Instant responses for repeated queries (< 1ms vs 2-5s)
- Reduced API costs (no LLM calls for cached queries)
- Lower carbon footprint (fewer compute resources)

FEATURES:
- In-memory cache with configurable TTL (time-to-live)
- Query normalization (case-insensitive, whitespace trimming)
- Cache hit/miss tracking
- Automatic expiration
- Cache statistics

USAGE:
    from src.utils.cache import QueryCache
    
    cache = QueryCache(ttl_seconds=3600)  # 1 hour TTL
    
    # Check cache
    cached_answer = cache.get(query)
    if cached_answer:
        return cached_answer
    
    # Generate new answer
    answer = pipeline.query(query)
    
    # Store in cache
    cache.set(query, answer)
"""

import time
import hashlib
from typing import Dict, Optional, Any
from collections import OrderedDict


class QueryCache:
    """
    Cache for RAG query results with TTL and size limits.
    
    ATTRIBUTES:
        ttl_seconds (int): Time-to-live for cache entries (seconds)
        max_size (int): Maximum number of entries to cache
        cache (OrderedDict): Stores cached results with timestamps
        stats (dict): Cache hit/miss statistics
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,  # 1 hour default
        max_size: int = 1000
    ):
        """
        Initialize query cache.
        
        PARAMETERS:
            ttl_seconds (int): How long to keep cache entries (seconds)
            max_size (int): Maximum cache size (entries)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent cache lookups.
        
        NORMALIZATION:
        - Convert to lowercase
        - Strip whitespace
        - Remove extra spaces
        
        EXAMPLE:
            "What is  AI? " -> "what is ai?"
        """
        return ' '.join(query.lower().strip().split())
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key from query.
        
        Uses MD5 hash for consistent key generation.
        """
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for query.
        
        PARAMETERS:
            query (str): User query
        
        RETURNS:
            Dict with cached result, or None if not found/expired
        
        SIDE EFFECTS:
        - Updates hit/miss statistics
        - Removes expired entries
        """
        cache_key = self._get_cache_key(query)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if expired
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                del self.cache[cache_key]
                self.stats['evictions'] += 1
                self.stats['misses'] += 1
                return None
            
            # Cache hit
            self.stats['hits'] += 1
            
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            
            return entry['result']
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    def set(self, query: str, result: Dict[str, Any]):
        """
        Store query result in cache.
        
        PARAMETERS:
            query (str): User query
            result (dict): Query result to cache
        
        SIDE EFFECTS:
        - Adds entry to cache
        - Evicts oldest entry if cache full
        - Updates statistics
        """
        cache_key = self._get_cache_key(query)
        
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self.cache.popitem(last=False)  # Remove oldest
            self.stats['evictions'] += 1
        
        # Store with timestamp
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        self.stats['sets'] += 1
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        RETURNS:
            Dict with:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - sets: Number of cache writes
            - evictions: Number of evicted entries
            - size: Current cache size
            - hit_rate: Cache hit rate (0-1)
        """
        total_queries = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_queries if total_queries > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'evictions': self.stats['evictions'],
            'size': len(self.cache),
            'hit_rate': hit_rate,
            'total_queries': total_queries
        }
    
    def __len__(self) -> int:
        """Return current cache size."""
        return len(self.cache)
    
    def __repr__(self) -> str:
        """String representation of cache."""
        stats = self.get_stats()
        return (
            f"QueryCache(size={stats['size']}/{self.max_size}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"ttl={self.ttl_seconds}s)"
        )


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Testing QueryCache...")
    
    cache = QueryCache(ttl_seconds=5, max_size=3)
    
    # Test set and get
    print("\n1. Testing set and get:")
    cache.set("What is AI?", {"answer": "AI is artificial intelligence"})
    result = cache.get("what is ai?")  # Different casing
    print(f"Cache hit (normalized): {result}")
    
    # Test statistics
    print("\n2. Cache statistics:")
    print(cache.get_stats())
    
    # Test miss
    print("\n3. Testing cache miss:")
    result = cache.get("What is ML?")
    print(f"Cache miss: {result}")
    
    # Test eviction
    print("\n4. Testing LRU eviction (max_size=3):")
    cache.set("Query 2", {"answer": "Answer 2"})
    cache.set("Query 3", {"answer": "Answer 3"})
    cache.set("Query 4", {"answer": "Answer 4"})  # Should evict first
    print(f"Cache size: {len(cache)}")
    print(f"First query still cached: {cache.get('What is AI?')}")
    
    # Test TTL expiration
    print("\n5. Testing TTL expiration (5s):")
    print("Waiting 6 seconds...")
    time.sleep(6)
    result = cache.get("Query 2")
    print(f"Expired entry: {result}")
    
    print("\n6. Final statistics:")
    print(cache)
