"""
Cost Tracking System
====================

Tracks API costs and token usage for RAG queries across multiple providers.

FEATURES:
- Token counting per query
- Cost estimation for Gemini, OpenAI, OpenRouter
- Per-provider cost tracking
- Aggregated statistics
- Cost breakdown by component (retrieval vs generation)

USAGE:
    from src.utils.cost_tracker import CostTracker
    
    tracker = CostTracker()
    
    # Track query
    tracker.track_query(
        provider='gemini',
        model='gemini-2.5-pro',
        input_tokens=500,
        output_tokens=200,
        query_time=2.5
    )
    
    # Get statistics
    stats = tracker.get_stats()
    print(f"Total cost: ${stats['total_cost']:.4f}")
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ==============================================================================
# PRICING DATA (per 1M tokens, as of 2024)
# ==============================================================================

PRICING = {
    'gemini': {
        'gemini-2.5-pro': {'input': 1.25, 'output': 5.00},
        'gemini-2.5-flash': {'input': 0.075, 'output': 0.30},
        'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
        'gemini-pro': {'input': 0.50, 'output': 1.50},
        'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
        'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
    },
    'openai': {
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-4': {'input': 30.00, 'output': 60.00},
        'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    },
    'openrouter': {
        # OpenRouter costs vary; using averages for free models
        'default': {'input': 0.00, 'output': 0.00},
    }
}


@dataclass
class QueryRecord:
    """Record of a single query's cost and metrics."""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    query_time: float
    query_text: Optional[str] = None


class CostTracker:
    """
    Track API costs and token usage across queries.
    
    ATTRIBUTES:
        records (List[QueryRecord]): All query records
        stats (dict): Aggregated statistics
    """
    
    def __init__(self):
        """Initialize cost tracker."""
        self.records: List[QueryRecord] = []
        self.start_time = time.time()
    
    def _estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for a query.
        
        PARAMETERS:
            provider (str): LLM provider (gemini, openai, openrouter)
            model (str): Model name
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
        
        RETURNS:
            float: Estimated cost in USD
        
        COST CALCULATION:
            cost = (input_tokens / 1M * input_price) + 
                   (output_tokens / 1M * output_price)
        """
        provider = provider.lower()
        
        # Get pricing
        if provider in PRICING:
            if model in PRICING[provider]:
                pricing = PRICING[provider][model]
            else:
                # Use default for provider
                pricing = PRICING[provider].get('default', {'input': 0, 'output': 0})
        else:
            pricing = {'input': 0, 'output': 0}
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        
        return input_cost + output_cost
    
    def track_query(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        query_time: float,
        query_text: Optional[str] = None
    ):
        """
        Track a query's cost and metrics.
        
        PARAMETERS:
            provider (str): LLM provider
            model (str): Model name
            input_tokens (int): Input token count
            output_tokens (int): Output token count
            query_time (float): Query processing time (seconds)
            query_text (str, optional): The query text
        """
        cost = self._estimate_cost(provider, model, input_tokens, output_tokens)
        
        record = QueryRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            query_time=query_time,
            query_text=query_text
        )
        
        self.records.append(record)
    
    def get_stats(self) -> Dict:
        """
        Get aggregated cost and usage statistics.
        
        RETURNS:
            Dict with:
            - total_queries: Number of queries tracked
            - total_cost: Total estimated cost (USD)
            - total_input_tokens: Total input tokens
            - total_output_tokens: Total output tokens
            - avg_query_time: Average query time (seconds)
            - cost_by_provider: Cost breakdown by provider
            - cost_by_model: Cost breakdown by model
            - queries_per_provider: Query count by provider
        """
        if not self.records:
            return {
                'total_queries': 0,
                'total_cost': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'avg_query_time': 0.0,
                'cost_by_provider': {},
                'cost_by_model': {},
                'queries_per_provider': {}
            }
        
        # Aggregate metrics
        total_cost = sum(r.cost for r in self.records)
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        avg_time = sum(r.query_time for r in self.records) / len(self.records)
        
        # Cost by provider
        cost_by_provider = {}
        for record in self.records:
            if record.provider not in cost_by_provider:
                cost_by_provider[record.provider] = 0.0
            cost_by_provider[record.provider] += record.cost
        
        # Cost by model
        cost_by_model = {}
        for record in self.records:
            if record.model not in cost_by_model:
                cost_by_model[record.model] = 0.0
            cost_by_model[record.model] += record.cost
        
        # Queries per provider
        queries_per_provider = {}
        for record in self.records:
            if record.provider not in queries_per_provider:
                queries_per_provider[record.provider] = 0
            queries_per_provider[record.provider] += 1
        
        return {
            'total_queries': len(self.records),
            'total_cost': total_cost,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_input + total_output,
            'avg_query_time': avg_time,
            'avg_cost_per_query': total_cost / len(self.records),
            'cost_by_provider': cost_by_provider,
            'cost_by_model': cost_by_model,
            'queries_per_provider': queries_per_provider,
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def get_recent_queries(self, n: int = 10) -> List[QueryRecord]:
        """Get the N most recent queries."""
        return self.records[-n:]
    
    def export_to_dict(self) -> List[Dict]:
        """Export all records to list of dicts for JSON serialization."""
        return [
            {
                'timestamp': r.timestamp.isoformat(),
                'provider': r.provider,
                'model': r.model,
                'input_tokens': r.input_tokens,
                'output_tokens': r.output_tokens,
                'cost': r.cost,
                'query_time': r.query_time,
                'query_text': r.query_text
            }
            for r in self.records
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"CostTracker("
            f"queries={stats['total_queries']}, "
            f"cost=${stats['total_cost']:.4f}, "
            f"tokens={stats['total_tokens']:,})"
        )


# ==============================================================================
# TOKEN ESTIMATION UTILITIES
# ==============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    APPROXIMATION:
    - English: ~4 characters per token
    - This is a rough estimate; actual tokenization varies by model
    
    For accurate counts, use tiktoken (OpenAI) or model-specific tokenizers.
    """
    return len(text) // 4


def estimate_query_tokens(query: str, contexts: List[Dict], answer: str) -> Dict[int, int]:
    """
    Estimate tokens for a RAG query.
    
    RETURNS:
        dict: {'input': input_tokens, 'output': output_tokens}
    """
    # Input: query + system prompt + contexts
    system_prompt_tokens = 100  # Approximate
    query_tokens = estimate_tokens(query)
    context_tokens = sum(estimate_tokens(c.get('text', '')) for c in contexts)
    
    input_tokens = system_prompt_tokens + query_tokens + context_tokens
    
    # Output: answer
    output_tokens = estimate_tokens(answer)
    
    return {
        'input': input_tokens,
        'output': output_tokens
    }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Testing CostTracker...")
    
    tracker = CostTracker()
    
    # Simulate queries
    print("\n1. Tracking queries:")
    tracker.track_query(
        provider='gemini',
        model='gemini-2.5-flash',
        input_tokens=500,
        output_tokens=200,
        query_time=1.5,
        query_text="What is AI?"
    )
    
    tracker.track_query(
        provider='openai',
        model='gpt-4o-mini',
        input_tokens=600,
        output_tokens=250,
        query_time=2.1,
        query_text="Explain transformers"
    )
    
    tracker.track_query(
        provider='gemini',
        model='gemini-2.5-pro',
        input_tokens=1000,
        output_tokens=500,
        query_time=3.2,
        query_text="Complex query"
    )
    
    # Get statistics
    print("\n2. Statistics:")
    stats = tracker.get_stats()
    
    print(f"Total queries: {stats['total_queries']}")
    print(f"Total cost: ${stats['total_cost']:.6f}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg query time: {stats['avg_query_time']:.2f}s")
    print(f"Avg cost per query: ${stats['avg_cost_per_query']:.6f}")
    
    print("\n3. Cost by provider:")
    for provider, cost in stats['cost_by_provider'].items():
        queries = stats['queries_per_provider'][provider]
        print(f"  {provider}: ${cost:.6f} ({queries} queries)")
    
    print("\n4. Cost by model:")
    for model, cost in stats['cost_by_model'].items():
        print(f"  {model}: ${cost:.6f}")
    
    print(f"\n5. Tracker: {tracker}")
