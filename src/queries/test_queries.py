# ==============================================================================
# FILE: test_queries.py
# PURPOSE: Define test queries for RAG system evaluation
# ==============================================================================

"""
This module contains a diverse set of test queries for evaluating the RAG system.

WHY TEST QUERIES?
- Consistent evaluation across retrieval, generation, and final system
- Cover different difficulty levels and question types
- Test various aspects: retrieval accuracy, generation quality, hallucination
- Provide ground truth for evaluation metrics

QUERY SELECTION RATIONALE:
We need queries that:
1. Cover all document types (PDF, DOCX, Excel)
2. Span difficulty levels (easy, medium, hard)
3. Test different capabilities (factual, technical, comparison, data)
4. Include edge cases to test robustness
5. Are realistic questions users might ask

USAGE:
    from src.queries.test_queries import QUERIES, get_all_queries
    
    # Get all queries
    queries = get_all_queries()
    
    # Get queries by category
    factual_queries = get_queries_by_category('factual')
    
    # Get queries by difficulty
    easy_queries = get_queries_by_difficulty('easy')
"""

# ==============================================================================
# TEST QUERIES DATASET
# ==============================================================================

QUERIES = [
    # =========================================================================
    # CATEGORY 1: FACTUAL/SIMPLE RETRIEVAL (Easy)
    # These test basic document retrieval and factual extraction
    # =========================================================================
    
    {
        'id': 1,
        'query': "What is the transformer architecture?",
        'category': 'factual',
        'difficulty': 'easy',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['encoder', 'decoder', 'attention', 'layers'],
        'rationale': """
            RATIONALE:
            - Tests basic concept retrieval from technical papers
            - Single-document, single-concept question
            - Clear answer should be in introduction/architecture sections
            - Good baseline for measuring retrieval accuracy
            - Expected to find answer in first few chunks of Attention paper
        """,
        'num_chunks_needed': 2  # Should need 1-2 chunks to answer well
    },
    
    {
        'id': 2,
        'query': "What are the main provisions of the EU AI Act?",
        'category': 'factual',
        'difficulty': 'easy',
        'expected_sources': ['EU AI Act Doc (1) (3).docx'],
        'expected_answer_contains': ['high-risk', 'prohibited', 'requirements', 'compliance'],
        'rationale': """
            RATIONALE:
            - Tests retrieval from regulatory/legal documents
            - Factual information retrieval (not technical)
            - Should be in summary sections of the document
            - Tests if system can handle legal/policy language
            - Different writing style than technical papers
        """,
        'num_chunks_needed': 3  # May need 2-3 chunks for comprehensive answer
    },
    
    {
        'id': 3,
        'query': "What was the inflation rate in 2020?",
        'category': 'data',
        'difficulty': 'easy',
        'expected_sources': ['Inflation Calculator.xlsx'],
        'expected_answer_contains': ['2020', 'inflation', 'rate', 'percent'],
        'rationale': """
            RATIONALE:
            - Tests retrieval from structured data (Excel)
            - Specific data point extraction
            - Tests if text conversion of Excel preserves data
            - Simple lookup query - no calculation needed
            - Good for testing data-handling pipeline
        """,
        'num_chunks_needed': 1  # Should be in one chunk
    },
    
    # =========================================================================
    # CATEGORY 2: TECHNICAL/DETAILED QUESTIONS (Medium)
    # These require understanding context and technical details
    # =========================================================================
    
    {
        'id': 4,
        'query': "How does self-attention mechanism work in transformers?",
        'category': 'technical',
        'difficulty': 'medium',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['query', 'key', 'value', 'attention weights', 'scaled dot-product'],
        'rationale': """
            RATIONALE:
            - Tests technical detail retrieval + explanation
            - Requires multiple chunks (definition + explanation + formula)
            - Tests if system can retrieve related concepts together
            - More complex than just factual retrieval
            - Good test of context window in retrieval
        """,
        'num_chunks_needed': 3  # Need multiple chunks for full explanation
    },
    
    {
        'id': 5,
        'query': "What is DeepSeek-R1 and how does it use reinforcement learning?",
        'category': 'technical',
        'difficulty': 'medium',
        'expected_sources': ['Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['reinforcement learning', 'reasoning', 'model', 'training'],
        'rationale': """
            RATIONALE:
            - Multi-part question (what + how)
            - Tests retrieval of method description
            - Requires understanding of ML terminology
            - Tests if system can connect concept to implementation
            - Good for testing answer coherence
        """,
        'num_chunks_needed': 4  # Abstract + method + results sections
    },
    
    {
        'id': 6,
        'query': "How has inflation changed from 1950 to 2021?",
        'category': 'data',
        'difficulty': 'medium',
        'expected_sources': ['Inflation Calculator.xlsx'],
        'expected_answer_contains': ['1950', '2021', 'inflation', 'change', 'increase'],
        'rationale': """
            RATIONALE:
            - Tests trend analysis from tabular data
            - Requires retrieving multiple data points
            - Tests if system can reason about numerical trends
            - More complex than single data point retrieval
            - Tests LLM's ability to interpret structured data
        """,
        'num_chunks_needed': 2  # May need chunks with different years
    },
    
    {
        'id': 7,
        'query': "What are the key innovations in the transformer model?",
        'category': 'technical',
        'difficulty': 'medium',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['multi-head attention', 'positional encoding', 'self-attention', 'feedforward'],
        'rationale': """
            RATIONALE:
            - Open-ended question requiring synthesis
            - Tests if retrieval captures multiple innovations
            - Requires understanding of what counts as "innovation"
            - Good test of semantic similarity (not just keyword match)
            - Tests LLM's ability to summarize technical content
        """,
        'num_chunks_needed': 4  # Multiple sections describing innovations
    },
    
    # =========================================================================
    # CATEGORY 3: COMPARISON/MULTI-DOCUMENT (Hard)
    # These require information from multiple sources or complex reasoning
    # =========================================================================
    
    {
        'id': 8,
        'query': "Compare the attention mechanism in transformers with traditional RNN approaches",
        'category': 'comparison',
        'difficulty': 'hard',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['attention', 'RNN', 'parallel', 'sequential', 'dependencies'],
        'rationale': """
            RATIONALE:
            - Comparison question requiring contrast
            - Tests if system can retrieve both concepts
            - Requires understanding advantages/disadvantages
            - Good test of multi-chunk reasoning
            - Tests LLM's ability to compare and contrast
            - May need background knowledge (RNNs) not in docs
        """,
        'num_chunks_needed': 5  # Multiple sections + introduction/background
    },
    
    {
        'id': 9,
        'query': "How do AI regulations address risks in machine learning systems?",
        'category': 'cross-document',
        'difficulty': 'hard',
        'expected_sources': ['EU AI Act Doc (1) (3).docx', 'Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['risk', 'AI', 'regulation', 'compliance', 'requirements'],
        'rationale': """
            RATIONALE:
            - Cross-document reasoning required
            - Tests if retrieval can find relevant info from multiple docs
            - Requires connecting regulatory concepts to technical ML
            - Tests semantic understanding (not just keyword match)
            - Good test of retrieval ranking across documents
            - Challenging - may require LLM to bridge concepts
        """,
        'num_chunks_needed': 4  # Chunks from both documents
    },
    
    # =========================================================================
    # CATEGORY 4: EDGE CASES/ROBUSTNESS (Testing Limits)
    # These test system robustness and hallucination detection
    # =========================================================================
    
    {
        'id': 10,
        'query': "What is the learning rate used in DeepSeek-R1 training?",
        'category': 'specific-technical',
        'difficulty': 'hard',
        'expected_sources': ['Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['learning rate', 'training'],  # May or may not have specific value
        'rationale': """
            RATIONALE:
            - Very specific technical detail
            - May or may not be in the document (tests hallucination)
            - If not in doc, system should say "not specified" not make it up
            - Good test of hallucination detection
            - Tests if retrieval can find very specific details
            - Tests LLM honesty - admitting when info is unavailable
        """,
        'num_chunks_needed': 2,  # If present, likely in methods section
        'hallucination_test': True  # Flag that this tests hallucination
    },
    
    # =========================================================================
    # CATEGORY 5: ADDITIONAL TECHNICAL DEPTH (Medium-Hard)
    # These test deeper understanding of technical concepts
    # =========================================================================
    
    {
        'id': 11,
        'query': "Explain the scaled dot-product attention mechanism and why scaling is important",
        'category': 'technical',
        'difficulty': 'medium',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['scaled dot-product', 'queries', 'keys', 'values', 'softmax', 'sqrt', 'dimension'],
        'rationale': """
            RATIONALE:
            - Tests deep technical understanding of core mechanism
            - Requires explanation of mathematical reasoning (why scaling)
            - Tests if system can retrieve formula + intuition
            - Good test of technical explanation quality
            - Tests understanding of attention mechanism details
        """,
        'num_chunks_needed': 3
    },
    
    {
        'id': 12,
        'query': "How does the Transformer handle sequential information without recurrence?",
        'category': 'technical',
        'difficulty': 'medium',
        'expected_sources': ['Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['positional encoding', 'sine', 'cosine', 'position', 'order', 'sequence'],
        'rationale': """
            RATIONALE:
            - Tests understanding of architectural innovation
            - Requires connecting positional encoding to sequence handling
            - Tests if system understands why positional encoding exists
            - Good test of conceptual understanding vs keyword matching
            - Tests ability to explain design decisions
        """,
        'num_chunks_needed': 3
    },
    
    {
        'id': 13,
        'query': "What types of AI systems are prohibited under the EU AI Act?",
        'category': 'factual',
        'difficulty': 'easy',
        'expected_sources': ['EU AI Act Doc (1) (3).docx'],
        'expected_answer_contains': ['prohibited', 'banned', 'social scoring', 'manipulation', 'exploitation'],
        'rationale': """
            RATIONALE:
            - Tests retrieval of specific regulatory restrictions
            - Simple factual question with clear answer
            - Tests extraction of list/enumeration from legal text
            - Good for testing precision of retrieval
            - Important real-world compliance question
        """,
        'num_chunks_needed': 2
    },
    
    {
        'id': 14,
        'query': "What are the penalties for non-compliance with the EU AI Act?",
        'category': 'factual',
        'difficulty': 'medium',
        'expected_sources': ['EU AI Act Doc (1) (3).docx'],
        'expected_answer_contains': ['fine', 'penalty', 'percentage', 'revenue', 'sanctions', 'enforcement'],
        'rationale': """
            RATIONALE:
            - Tests extraction of enforcement details
            - May require retrieving specific numbers/percentages
            - Tests if system can handle legal terminology
            - Good for testing structured information retrieval
            - Practical compliance information
        """,
        'num_chunks_needed': 2
    },
    
    {
        'id': 15,
        'query': "What are the key stages in DeepSeek-R1's training methodology?",
        'category': 'technical',
        'difficulty': 'easy',
        'expected_sources': ['Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['training', 'stage', 'reinforcement learning', 'fine-tuning', 'methodology'],
        'rationale': """
            RATIONALE:
            - Tests understanding of multi-stage training pipeline
            - Simple structured information (stages/steps)
            - Tests if system can identify process flow
            - Good for testing technical documentation retrieval
            - Tests extraction of methodology information
        """,
        'num_chunks_needed': 3
    },
    
    {
        'id': 16,
        'query': "How does DeepSeek-R1 compare to other models on reasoning benchmarks?",
        'category': 'comparison',
        'difficulty': 'medium',
        'expected_sources': ['Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['benchmark', 'performance', 'accuracy', 'comparison', 'evaluation', 'score'],
        'rationale': """
            RATIONALE:
            - Tests extraction of quantitative performance data
            - Comparison question requiring multiple data points
            - Tests if system can handle tables/results sections
            - Good for testing structured data in PDFs
            - Tests understanding of evaluation metrics
        """,
        'num_chunks_needed': 3
    },
    
    {
        'id': 17,
        'query': "Explain the distillation process used in DeepSeek-R1 and its benefits",
        'category': 'technical',
        'difficulty': 'hard',
        'expected_sources': ['Deepseek-r1 (1).pdf'],
        'expected_answer_contains': ['distillation', 'teacher', 'student', 'knowledge transfer', 'efficiency', 'smaller'],
        'rationale': """
            RATIONALE:
            - Tests understanding of advanced ML technique
            - Requires explanation of process + benefits
            - Multi-part question (what + why)
            - Tests if system can connect technique to outcomes
            - Good test of technical explanation depth
        """,
        'num_chunks_needed': 4
    },
    
    # =========================================================================
    # CATEGORY 6: DATA ANALYSIS AND COMPARISON (Easy-Medium)
    # These test numerical data handling and comparison
    # =========================================================================
    
    {
        'id': 18,
        'query': "What was the highest inflation rate recorded in the dataset and in which year?",
        'category': 'data',
        'difficulty': 'easy',
        'expected_sources': ['Inflation Calculator.xlsx'],
        'expected_answer_contains': ['highest', 'maximum', 'peak', 'year', 'inflation', 'rate'],
        'rationale': """
            RATIONALE:
            - Tests simple aggregation (finding maximum)
            - Requires both value and associated year
            - Tests if Excel data is properly indexed
            - Good for testing data lookup accuracy
            - Simple but important analytical question
        """,
        'num_chunks_needed': 1
    },
    
    {
        'id': 19,
        'query': "Compare the inflation rates between 2019 and 2022",
        'category': 'comparison',
        'difficulty': 'medium',
        'expected_sources': ['Inflation Calculator.xlsx'],
        'expected_answer_contains': ['2019', '2022', 'compare', 'difference', 'inflation', 'rate'],
        'rationale': """
            RATIONALE:
            - Tests year-over-year comparison
            - Requires retrieving multiple data points
            - Tests if system can perform simple calculations
            - Good for testing temporal data handling
            - Tests LLM's ability to interpret numerical differences
        """,
        'num_chunks_needed': 2
    },
    
    # =========================================================================
    # CATEGORY 7: CROSS-DOCUMENT SYNTHESIS (Hard)
    # These test multi-document reasoning and synthesis
    # =========================================================================
    
    {
        'id': 20,
        'query': "How might the EU AI Act's transparency requirements apply to transformer-based language models?",
        'category': 'synthesis',
        'difficulty': 'hard',
        'expected_sources': ['EU AI Act Doc (1) (3).docx', 'Attention_is_all_you_need (1) (3).pdf'],
        'expected_answer_contains': ['transparency', 'explainability', 'AI Act', 'transformer', 'requirements', 'compliance'],
        'rationale': """
            RATIONALE:
            - Complex cross-document reasoning
            - Requires connecting regulatory concepts to technical architecture
            - Tests ability to synthesize information from different domains
            - Good test of semantic understanding across document types
            - Tests LLM's ability to apply regulatory framework to technical system
            - Realistic real-world question (compliance for ML systems)
        """,
        'num_chunks_needed': 5,  # Multiple chunks from both documents
        'hallucination_test': True  # May require inference beyond document content
    },
]

# Backward-compatible alias expected by some scripts
# Some demo scripts import TEST_QUERIES; keep it in sync with QUERIES.
TEST_QUERIES = QUERIES

# Explicit exports for clarity
__all__ = [
    'QUERIES',
    'TEST_QUERIES',
    'get_all_queries',
    'get_queries_by_category',
    'get_queries_by_difficulty',
    'get_hallucination_test_queries',
    'get_query_by_id',
    'print_query_summary',
]


# ==============================================================================
# QUERY STATISTICS AND METADATA
# ==============================================================================

QUERY_STATS = {
    'total_queries': len(QUERIES),  # Now 20 queries
    'by_category': {
        'factual': 4,           # Q2, Q13, Q14, Q15
        'data': 4,              # Q3, Q6, Q18, Q19
        'technical': 7,         # Q4, Q7, Q11, Q12, Q15, Q16, Q17
        'comparison': 3,        # Q8, Q16, Q19
        'cross-document': 1,    # Q9
        'specific-technical': 1, # Q10
        'synthesis': 1          # Q20
    },
    'by_difficulty': {
        'easy': 6,   # Q1, Q2, Q3, Q13, Q15, Q18
        'medium': 8, # Q4, Q5, Q6, Q7, Q11, Q12, Q14, Q16, Q19
        'hard': 6    # Q8, Q9, Q10, Q17, Q20
    },
    'by_document_type': {
        'attention_paper': 5,    # Q1, Q4, Q7, Q8, Q11, Q12, Q20
        'eu_ai_act': 5,          # Q2, Q9, Q13, Q14, Q20
        'deepseek': 5,           # Q5, Q9, Q10, Q15, Q16, Q17
        'inflation_excel': 4,    # Q3, Q6, Q18, Q19
        'multi_doc': 2           # Q9, Q20
    },
    'document_coverage': {
        'Attention_is_all_you_need (1) (3).pdf': 6,
        'EU AI Act Doc (1) (3).docx': 5,
        'Deepseek-r1 (1).pdf': 6,
        'Inflation Calculator.xlsx': 4
    }
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_all_queries():
    """
    Get all test queries.
    
    RETURNS:
        list[dict]: All query objects with metadata
        
    EXAMPLE:
        queries = get_all_queries()
        for q in queries:
            print(f"Query {q['id']}: {q['query']}")
    """
    return QUERIES


def get_queries_by_category(category):
    """
    Filter queries by category.
    
    CATEGORIES:
        - 'factual': Basic factual questions
        - 'technical': Technical detail questions
        - 'data': Data/numerical questions
        - 'comparison': Comparison questions
        - 'cross-document': Multi-document questions
        - 'specific-technical': Very specific technical details
    
    PARAMETERS:
        category (str): Category to filter by
        
    RETURNS:
        list[dict]: Queries matching the category
        
    EXAMPLE:
        factual_queries = get_queries_by_category('factual')
    """
    return [q for q in QUERIES if q['category'] == category]


def get_queries_by_difficulty(difficulty):
    """
    Filter queries by difficulty level.
    
    DIFFICULTY LEVELS:
        - 'easy': Single-document, direct answer
        - 'medium': Multi-chunk or complex retrieval
        - 'hard': Multi-document or complex reasoning
    
    PARAMETERS:
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        
    RETURNS:
        list[dict]: Queries matching the difficulty
        
    EXAMPLE:
        easy_queries = get_queries_by_difficulty('easy')
    """
    return [q for q in QUERIES if q['difficulty'] == difficulty]


def get_hallucination_test_queries():
    """
    Get queries specifically designed to test hallucination.
    
    These queries may ask for information not in the documents.
    The system should admit when it doesn't know, not make up answers.
    
    RETURNS:
        list[dict]: Queries flagged as hallucination tests
    """
    return [q for q in QUERIES if q.get('hallucination_test', False)]


def print_query_summary():
    """
    Print a summary of all test queries.
    
    Shows overview of query distribution by category, difficulty, etc.
    Useful for documentation and understanding test coverage.
    """
    print("="*80)
    print("TEST QUERIES SUMMARY")
    print("="*80)
    
    print(f"\nTotal Queries: {QUERY_STATS['total_queries']}")
    
    print("\n" + "-"*80)
    print("BY CATEGORY:")
    print("-"*80)
    for category, count in QUERY_STATS['by_category'].items():
        print(f"  {category:20s}: {count} queries")
    
    print("\n" + "-"*80)
    print("BY DIFFICULTY:")
    print("-"*80)
    for difficulty, count in QUERY_STATS['by_difficulty'].items():
        print(f"  {difficulty:20s}: {count} queries")
    
    print("\n" + "-"*80)
    print("BY DOCUMENT TYPE:")
    print("-"*80)
    for doc_type, count in QUERY_STATS['by_document_type'].items():
        print(f"  {doc_type:20s}: {count} queries")
    
    print("\n" + "="*80)
    print("QUERY LIST:")
    print("="*80)
    for q in QUERIES:
        print(f"\n[{q['id']}] {q['query']}")
        print(f"    Category: {q['category']} | Difficulty: {q['difficulty']}")
        print(f"    Expected Sources: {', '.join(q['expected_sources'])}")
        print(f"    Chunks Needed: {q['num_chunks_needed']}")


def get_query_by_id(query_id):
    """
    Get a specific query by its ID.
    
    PARAMETERS:
        query_id (int): Query ID (1-10)
        
    RETURNS:
        dict: Query object, or None if not found
        
    EXAMPLE:
        query = get_query_by_id(1)
        print(query['query'])
    """
    for q in QUERIES:
        if q['id'] == query_id:
            return q
    return None


# ==============================================================================
# SELF-TEST / DEMO BLOCK
# ==============================================================================

if __name__ == "__main__":
    """
    Test the query module and show overview.
    Run this file directly to see all queries:
        python src/queries/test_queries.py
    """
    
    print("="*80)
    print("TESTING: test_queries.py")
    print("="*80)
    
    # Show summary
    print_query_summary()
    
    # Test filtering functions
    print("\n" + "="*80)
    print("TESTING FILTER FUNCTIONS")
    print("="*80)
    
    print("\n--- Easy Queries ---")
    easy = get_queries_by_difficulty('easy')
    for q in easy:
        print(f"  [{q['id']}] {q['query']}")
    
    print("\n--- Factual Queries ---")
    factual = get_queries_by_category('factual')
    for q in factual:
        print(f"  [{q['id']}] {q['query']}")
    
    print("\n--- Hallucination Test Queries ---")
    halluc = get_hallucination_test_queries()
    for q in halluc:
        print(f"  [{q['id']}] {q['query']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
